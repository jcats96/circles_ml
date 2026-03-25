"""
tests/test_training_jobs.py – tests for web/training_jobs.py
"""

import os
import queue
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.training_jobs import (
    EpochMetric,
    JobStatus,
    TrainingJob,
    create_job,
    get_job,
    list_jobs,
)


class TestEpochMetric:
    def test_fields_stored(self):
        m = EpochMetric(model="CNN", epoch=1, loss=0.5, mae=0.3, val_loss=0.6, val_mae=0.4)
        assert m.model == "CNN"
        assert m.epoch == 1
        assert m.loss == 0.5
        assert m.mae == 0.3
        assert m.val_loss == 0.6
        assert m.val_mae == 0.4

    def test_val_fields_can_be_none(self):
        m = EpochMetric(model="Dense", epoch=2, loss=0.1, mae=0.2, val_loss=None, val_mae=None)
        assert m.val_loss is None
        assert m.val_mae is None


class TestTrainingJob:
    def test_initial_status_is_pending(self):
        job = TrainingJob(job_id="test-001")
        assert job.status == JobStatus.PENDING

    def test_add_metric_appends(self):
        job = TrainingJob(job_id="test-002")
        m = EpochMetric("Dense", 1, 0.5, 0.3, None, None)
        job.add_metric(m)
        assert len(job.metrics) == 1
        assert job.metrics[0] is m

    def test_add_metric_notifies_subscribers(self):
        job = TrainingJob(job_id="test-003")
        q = queue.Queue()
        job.subscribe(q)
        m = EpochMetric("CNN", 1, 0.2, 0.1, None, None)
        job.add_metric(m)
        received = q.get_nowait()
        assert received is m

    def test_unsubscribe_removes_queue(self):
        job = TrainingJob(job_id="test-004")
        q = queue.Queue()
        job.subscribe(q)
        job.unsubscribe(q)
        m = EpochMetric("Dense", 1, 0.2, 0.1, None, None)
        job.add_metric(m)
        assert q.empty()

    def test_finish_sets_status_completed(self):
        job = TrainingJob(job_id="test-005")
        job.finish({"result": "ok"})
        assert job.status == JobStatus.COMPLETED
        assert job.summary == {"result": "ok"}

    def test_finish_sends_sentinel_to_subscribers(self):
        job = TrainingJob(job_id="test-006")
        q = queue.Queue()
        job.subscribe(q)
        job.finish({})
        sentinel = q.get_nowait()
        assert sentinel is None

    def test_fail_sets_status_failed(self):
        job = TrainingJob(job_id="test-007")
        job.fail("something went wrong")
        assert job.status == JobStatus.FAILED
        assert job.error == "something went wrong"

    def test_fail_sends_sentinel(self):
        job = TrainingJob(job_id="test-008")
        q = queue.Queue()
        job.subscribe(q)
        job.fail("err")
        assert q.get_nowait() is None

    def test_cancel_sets_status_cancelled(self):
        job = TrainingJob(job_id="test-009")
        job.status = JobStatus.RUNNING
        job.cancel()
        assert job.status == JobStatus.CANCELLED

    def test_cancel_only_cancels_running_jobs(self):
        job = TrainingJob(job_id="test-010")
        job.status = JobStatus.COMPLETED
        job.cancel()
        assert job.status == JobStatus.COMPLETED  # unchanged

    def test_cancel_sends_sentinel(self):
        job = TrainingJob(job_id="test-011")
        job.status = JobStatus.RUNNING
        q = queue.Queue()
        job.subscribe(q)
        job.cancel()
        assert q.get_nowait() is None

    def test_multiple_subscribers_all_notified(self):
        job = TrainingJob(job_id="test-012")
        queues = [queue.Queue() for _ in range(5)]
        for q in queues:
            job.subscribe(q)
        job.finish({})
        for q in queues:
            assert q.get_nowait() is None


class TestRegistry:
    def test_create_job_returns_job(self):
        job = create_job({"epochs": 5})
        assert job.job_id.startswith("train_")
        assert job.config == {"epochs": 5}

    def test_get_job_returns_created_job(self):
        job = create_job({})
        retrieved = get_job(job.job_id)
        assert retrieved is job

    def test_get_job_returns_none_for_unknown(self):
        assert get_job("nonexistent-job-id") is None

    def test_list_jobs_includes_created(self):
        job = create_job({"epochs": 1})
        jobs = list_jobs()
        assert job in jobs

    def test_two_jobs_have_different_ids(self):
        j1 = create_job({})
        j2 = create_job({})
        assert j1.job_id != j2.job_id
