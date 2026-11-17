"""Regression tests for the Venus E 3.0 v401 GetData telemetry."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.marstek.coordinator import MarstekDataUpdateCoordinator
from custom_components.marstek.firmware_profile import resolve_firmware_profile
from custom_components.marstek.pymarstek.data_parser import parse_marstek_data_response
from custom_components.marstek.pymarstek.udp import MarstekUDPClient
from tests.conftest import create_mock_client, patch_marstek_integration


@pytest.mark.parametrize(
    ("model", "version", "supported"),
    [
        ("VenusE 3.0", 401, True),
        ("VNSE3-0", 401, True),
        ("VenusE 3.0", 150, False),
        ("VenusE 3.0", 151, True),
        ("VenusE 3.0", 402, True),
        ("VenusE 3.0", None, False),
        ("VenusE", 401, False),
        ("VenusC", 401, False),
        ("VenusD", 401, False),
        ("VNSEM-0", 401, False),
    ],
)
def test_get_data_firmware_gate(model: str, version: int | None, supported: bool) -> None:
    """Introduce no GetData traffic to profiles without a verified reply."""
    profile = resolve_firmware_profile(model, version)
    assert profile.supports_marstek_data is supported
    if supported:
        previous = resolve_firmware_profile(model, 150)
        assert profile.setup_reload_signature != previous.setup_reload_signature


@pytest.mark.parametrize("raw_state", [0, 4, True, "3", [3], None])
def test_get_data_unknown_battery_state_preserves_power(raw_state: Any) -> None:
    """An unknown state cannot become a valid enum at HA's sensor boundary."""
    result = parse_marstek_data_response(
        {"result": {"bat_state": raw_state, "grid_pow": 0, "off_grid_pow": 11, "inv_pow": 11}}
    )
    assert result == {"grid_power": 0, "offgrid_power": 11, "inverter_power": 11}


@pytest.mark.parametrize("raw_power", [None, True, "unknown", float("inf")])
def test_get_data_incomplete_power_snapshot(raw_power: Any) -> None:
    """An invalid reading must not combine values from different replies."""
    assert parse_marstek_data_response(
        {"result": {"bat_state": 1, "grid_pow": raw_power, "off_grid_pow": 11, "inv_pow": 11}}
    ) == {"battery_status": "idle"}


@pytest.mark.parametrize(
    ("raw_state", "grid", "outlet", "inverter", "battery_status"),
    [
        (1, -116, 116, 0, "idle"),  # Mains bypass under load.
        (1, 0, 11, 11, "idle"),  # Light load on battery stays below the state threshold.
        (3, 0, 41, 41, "discharging"),
        (2, -1103, 34, -1069, "charging"),
    ],
)
async def test_get_data_polled_and_preferred_on_v401(
    raw_state: int, grid: int, outlet: int, inverter: int, battery_status: str
) -> None:
    """Use all three powers from GetData and leave the legacy power estimate in place."""
    client = MarstekUDPClient()
    client._loop = MagicMock()
    client._loop.time.return_value = 1000.0
    commands: list[dict[str, Any]] = []

    async def reply(message: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        command = json.loads(message)
        commands.append(command)
        return {
            "result": {
                "ES.GetMode": {"id": 0, "mode": "Manual", "bat_soc": 99},
                "ES.GetStatus": {"pv_power": 0, "ongrid_power": 0, "offgrid_power": 115},
                "Marstek.GetData": {
                    "bat_state": raw_state,
                    "grid_pow": grid,
                    "off_grid_pow": outlet,
                    "inv_pow": inverter,
                },
            }[command["method"]]
        }

    with patch.object(client, "send_request", side_effect=reply):
        for version in (401, 150):
            result = await client.get_device_status(
                "192.0.2.1",
                delay_between_requests=0,
                include_em=False,
                include_pv=False,
                include_wifi=False,
                include_bat=False,
                profile=resolve_firmware_profile("VenusE 3.0", version),
            )
            assert result["battery_power"] == 115
            assert result["device_mode"] == "manual"
            if version == 401:
                assert result["battery_status"] == battery_status
                assert result["grid_power"] == grid
                assert result["offgrid_power"] == outlet
                assert result["inverter_power"] == inverter
            else:
                assert result["battery_status"] == "discharging"
                assert "grid_power" not in result
                assert "inverter_power" not in result

    get_data_commands = [command for command in commands if command["method"] == "Marstek.GetData"]
    assert len(get_data_commands) == 1
    assert get_data_commands[0]["params"] == {"id": 0}


@pytest.mark.parametrize("get_data_reply", [TimeoutError(), {"result": {"bat_state": True}}])
async def test_get_data_missing_does_not_republish_previous_readings(
    get_data_reply: dict[str, Any] | TimeoutError,
) -> None:
    """An ES reply still updates legacy sensors when the GetData sample is missing."""
    client = MarstekUDPClient()
    client._loop = MagicMock()
    client._loop.time.return_value = 1000.0
    client.send_request = AsyncMock(
        side_effect=[
            {"result": {"id": 0, "mode": "Manual"}},
            {"result": {"pv_power": 0, "ongrid_power": 0, "offgrid_power": 120}},
            get_data_reply,
        ]
    )
    result = await client.get_device_status(
        "192.0.2.1",
        delay_between_requests=0,
        include_em=False,
        include_pv=False,
        include_wifi=False,
        include_bat=False,
        previous_status={"battery_status": "idle", "grid_power": -116, "inverter_power": 0},
        profile=resolve_firmware_profile("VenusE 3.0", 401),
    )
    assert result["has_fresh_data"]
    assert result["battery_power"] == 120
    assert result["offgrid_power"] == 120
    assert result["battery_status"] is None
    assert result["grid_power"] is None
    assert result["inverter_power"] is None


async def test_get_data_sensors_recover_and_clear_after_timeout(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """HA registers the sensors through a missing first sample and preserves their IDs."""
    mock_config_entry.add_to_hass(hass)
    hass.config_entries.async_update_entry(
        mock_config_entry,
        data={**mock_config_entry.data, "device_type": "VenusE 3.0", "version": 401},
    )
    missing_sample = {
        "device_mode": "Manual",
        "battery_soc": 99,
        "battery_status": None,
        "grid_power": None,
        "inverter_power": None,
    }
    sample = {**missing_sample, "battery_status": "idle", "grid_power": -116, "inverter_power": 0}
    client = create_mock_client(status=missing_sample)
    client.get_device_status = AsyncMock(side_effect=[missing_sample, sample, TimeoutError()])

    with patch_marstek_integration(client=client):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        entity_ids = (
            "sensor.venus_e_3_0_battery_status",
            "sensor.venus_e_3_0_grid_power",
            "sensor.venus_e_3_0_inverter_power",
        )
        assert [state_value(hass, entity_id) for entity_id in entity_ids] == ["unknown"] * 3

        coordinator: MarstekDataUpdateCoordinator = mock_config_entry.runtime_data.coordinator
        await coordinator.async_refresh()
        assert [state_value(hass, entity_id) for entity_id in entity_ids] == ["idle", "-116", "0"]

        await coordinator.async_refresh()
        assert [state_value(hass, entity_id) for entity_id in entity_ids] == ["unknown"] * 3
        assert state_value(hass, "sensor.venus_e_3_0_battery_level") == "99"


def state_value(hass: HomeAssistant, entity_id: str) -> str:
    """Return the state of a sensor that must exist."""
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state
