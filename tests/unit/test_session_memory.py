"""Tests for the shared process-local session size and access ledger."""

from uuid import UUID, uuid4

import pytest
from tests.support.fake_clock import FakeClock

from agentic_eval_toyapp.session_memory import (
    SESSION_ALLOWANCE_BYTES,
    SessionMemoryLedger,
    report_footprint,
    turn_footprint,
)


def test_turn_footprint_counts_both_messages() -> None:
    """A turn costs the length of its user and assistant text."""
    assert turn_footprint("abc", "de") == 5


def test_report_footprint_counts_original_bytes_alone() -> None:
    """Original bytes are counted once, independently of the text."""
    assert report_footprint(b"x" * 1_000, "", ()) == 1_000


def test_report_footprint_counts_extracted_text_once_without_pages() -> None:
    """A text report has no pages, so its text is paid for once."""
    assert report_footprint(b"", "y" * 300, ()) == 300


def test_report_footprint_counts_pdf_page_text_on_top_of_extracted_text() -> None:
    """A PDF keeps its joined text and every page's text, so both count."""
    pages = ("a" * 40, "b" * 60)

    assert report_footprint(b"", "\n\n".join(pages), pages) == 102 + 100


def test_report_footprint_sums_every_piece() -> None:
    """Original bytes, extracted text and pages add up."""
    assert report_footprint(b"12345", "abc", ("ab", "c")) == 5 + 3 + 3


def test_empty_session_costs_exactly_the_allowance() -> None:
    """An empty session is never free, so empty-session floods use budget."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    session_id = uuid4()

    ledger.track(session_id)

    footprint = ledger.footprint(session_id)
    assert footprint is not None
    assert footprint.session_bytes == SESSION_ALLOWANCE_BYTES
    assert footprint.report_bytes == 0
    assert footprint.total_bytes == SESSION_ALLOWANCE_BYTES
    assert ledger.total_bytes == SESSION_ALLOWANCE_BYTES


def test_empty_ledger_has_no_sessions_and_no_bytes() -> None:
    """A fresh ledger tracks nothing."""
    ledger = SessionMemoryLedger(clock=FakeClock())

    assert ledger.total_bytes == 0
    assert ledger.idle_order() == ()
    assert ledger.footprint(uuid4()) is None


def test_adjust_updates_session_and_report_bytes_and_the_total() -> None:
    """Deltas accumulate per session and in the total."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    first, second = uuid4(), uuid4()
    ledger.track(first)
    ledger.track(second)

    ledger.adjust(first, session_delta=10, report_delta=500)
    ledger.adjust(second, report_delta=7)
    ledger.adjust(first, report_delta=-200)

    first_footprint = ledger.footprint(first)
    assert first_footprint is not None
    assert first_footprint.session_bytes == SESSION_ALLOWANCE_BYTES + 10
    assert first_footprint.report_bytes == 300
    assert ledger.total_bytes == 2 * SESSION_ALLOWANCE_BYTES + 10 + 300 + 7


def test_adjust_rejects_a_negative_estimate_and_changes_nothing() -> None:
    """Drift below zero is a bug, so it raises instead of being clamped."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    session_id = uuid4()
    ledger.track(session_id)
    ledger.adjust(session_id, report_delta=5)

    with pytest.raises(ValueError, match="negative"):
        ledger.adjust(session_id, report_delta=-6)
    with pytest.raises(ValueError, match="negative"):
        ledger.adjust(session_id, session_delta=-1)

    assert ledger.total_bytes == SESSION_ALLOWANCE_BYTES + 5


def test_track_twice_is_rejected() -> None:
    """Re-tracking would double-count the allowance."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    session_id = uuid4()
    ledger.track(session_id)

    with pytest.raises(ValueError, match="already tracked"):
        ledger.track(session_id)
    assert ledger.total_bytes == SESSION_ALLOWANCE_BYTES


def test_untracked_sessions_are_ignored() -> None:
    """Updates for an untracked ID, such as a demo session, change nothing."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    unknown = uuid4()

    ledger.touch(unknown)
    ledger.adjust(unknown, session_delta=100, report_delta=100)

    assert ledger.footprint(unknown) is None
    assert ledger.total_bytes == 0
    assert ledger.idle_order() == ()


def test_idle_order_is_creation_order_until_a_session_is_touched() -> None:
    """Longest-idle first; a touch moves a session to the end."""
    clock = FakeClock()
    ledger = SessionMemoryLedger(clock=clock)
    ids: list[UUID] = [uuid4(), uuid4(), uuid4()]
    for session_id in ids:
        ledger.track(session_id)
        clock.advance(1)

    assert ledger.idle_order() == tuple(ids)

    ledger.touch(ids[0])

    assert ledger.idle_order() == (ids[1], ids[2], ids[0])
    touched = ledger.footprint(ids[0])
    assert touched is not None
    assert touched.last_access == 103.0


def test_sessions_tracked_at_the_same_instant_keep_insertion_order() -> None:
    """Equal timestamps break ties by insertion order, not by ID."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    ids = [UUID(int=3), UUID(int=1), UUID(int=2)]
    for session_id in ids:
        ledger.track(session_id)

    assert ledger.idle_order() == tuple(ids)


def test_adjust_does_not_count_as_an_access() -> None:
    """Only touch moves a session in the idle order."""
    clock = FakeClock()
    ledger = SessionMemoryLedger(clock=clock)
    first, second = uuid4(), uuid4()
    ledger.track(first)
    ledger.track(second)
    clock.advance(5)

    ledger.adjust(first, report_delta=10)

    assert ledger.idle_order() == (first, second)
    footprint = ledger.footprint(first)
    assert footprint is not None
    assert footprint.last_access == 100.0
