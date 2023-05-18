"""This module contains the type definitions for the parsed json structure."""

from typing import Literal, TypedDict

import pandas as pd

IDNumberDict = dict[Literal["ct", "t"], dict[str, Literal["1", "2", "3", "4", "5"]]]

DictInitialized = dict[Literal["ct", "t"], bool]


class AllowedForbidden(TypedDict):
    """Holds information about which values are allowed or forbidden."""

    Allowed: list[str]
    Forbidden: list[str]


class EquipSpecification(TypedDict):
    """Holds information about equipement specifications."""

    Kill: list[str]
    CT: AllowedForbidden
    T: AllowedForbidden


class PositionSpecification(TypedDict):
    """Holds information about which positions are of interest."""

    CT: list[str]
    T: list[str]


class WeaponsClassesSwitch(TypedDict):
    """Holds information about whether to filter by weapons or classes."""

    CT: Literal["weapons", "classes"]
    T: Literal["weapons", "classes"]
    Kill: Literal["weapons", "classes"]


class TimeSpecification(TypedDict):
    """Holds information about the time the fights should have occured in."""

    start: int
    end: int


class FightSpecification(TypedDict):
    """Holds information about fights to parse."""

    map_name: str
    weapons: EquipSpecification
    classes: EquipSpecification
    positions: PositionSpecification
    use_weapons_classes: WeaponsClassesSwitch
    times: TimeSpecification


class QueryResult(TypedDict):
    """Holds information about result for querying a specific fight situation."""

    situations_found: int
    ct_win_percentage: tuple[float, float, float]


class TickInformation(TypedDict):
    """Holds information about tickrate and freezetimeendtick."""

    tickRate: int
    freezeTimeEndTick: int


class RoundPositions(TypedDict):
    """Holds information about player positions during a round."""

    Tick: list[int]
    token: list[str]
    interpolated: list[Literal[0, 1]]
    CTtoken: list[str]
    CTPlayer1Alive: list[Literal[0, 1]]
    CTPlayer1Name: list[str]
    CTPlayer1x: list[float]
    CTPlayer1y: list[float]
    CTPlayer1z: list[float]
    CTPlayer1Area: list[int]
    CTPlayer2Alive: list[Literal[0, 1]]
    CTPlayer2Name: list[str]
    CTPlayer2x: list[float]
    CTPlayer2y: list[float]
    CTPlayer2z: list[float]
    CTPlayer2Area: list[int]
    CTPlayer3Alive: list[Literal[0, 1]]
    CTPlayer3Name: list[str]
    CTPlayer3x: list[float]
    CTPlayer3y: list[float]
    CTPlayer3z: list[float]
    CTPlayer3Area: list[int]
    CTPlayer4Alive: list[Literal[0, 1]]
    CTPlayer4Name: list[str]
    CTPlayer4x: list[float]
    CTPlayer4y: list[float]
    CTPlayer4z: list[float]
    CTPlayer4Area: list[int]
    CTPlayer5Alive: list[Literal[0, 1]]
    CTPlayer5Name: list[str]
    CTPlayer5x: list[float]
    CTPlayer5y: list[float]
    CTPlayer5z: list[float]
    CTPlayer5Area: list[int]
    Ttoken: list[str]
    TPlayer1Alive: list[Literal[0, 1]]
    TPlayer1Name: list[str]
    TPlayer1x: list[float]
    TPlayer1y: list[float]
    TPlayer1z: list[float]
    TPlayer1Area: list[int]
    TPlayer2Alive: list[Literal[0, 1]]
    TPlayer2Name: list[str]
    TPlayer2x: list[float]
    TPlayer2y: list[float]
    TPlayer2z: list[float]
    TPlayer2Area: list[int]
    TPlayer3Alive: list[Literal[0, 1]]
    TPlayer3Name: list[str]
    TPlayer3x: list[float]
    TPlayer3y: list[float]
    TPlayer3z: list[float]
    TPlayer3Area: list[int]
    TPlayer4Alive: list[Literal[0, 1]]
    TPlayer4Name: list[str]
    TPlayer4x: list[float]
    TPlayer4y: list[float]
    TPlayer4z: list[float]
    TPlayer4Area: list[int]
    TPlayer5Alive: list[Literal[0, 1]]
    TPlayer5Name: list[str]
    TPlayer5x: list[float]
    TPlayer5y: list[float]
    TPlayer5z: list[float]
    TPlayer5Area: list[int]


class PositionDataset(TypedDict):
    """Holds information about the rounds of a game."""

    MatchID: list[str]
    MapName: list[str]
    Round: list[int]
    Winner: list[Literal[0, 1]]
    position_df: list[pd.DataFrame]
