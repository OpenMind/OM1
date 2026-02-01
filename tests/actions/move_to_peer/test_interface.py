"""Tests for move_to_peer interface (MoveToPeerAction, MoveToPeerInput, MoveToPeer)."""

import pytest

from actions.move_to_peer.interface import (
    MoveToPeer,
    MoveToPeerAction,
    MoveToPeerInput,
)


def test_move_to_peer_action_enum_values():
    """MoveToPeerAction has expected idle and navigate values."""
    assert MoveToPeerAction.IDLE == "idle"
    assert MoveToPeerAction.NAVIGATE == "navigate"


def test_move_to_peer_input_dataclass():
    """MoveToPeerInput accepts action and is usable as protocol."""
    inp = MoveToPeerInput(action=MoveToPeerAction.IDLE)
    assert inp.action == MoveToPeerAction.IDLE
    inp_nav = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
    assert inp_nav.action == MoveToPeerAction.NAVIGATE


def test_move_to_peer_interface_structure():
    """MoveToPeer interface has input and output of type MoveToPeerInput."""
    inp = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
    iface = MoveToPeer(input=inp, output=inp)
    assert iface.input is inp
    assert iface.output is inp
    assert iface.input.action == MoveToPeerAction.NAVIGATE
