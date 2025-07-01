"""Unit tests for the main 'record' module."""

import json
import os

import arrow
from click import get_app_dir
import pytest
import requests

from record import Record, RecordError
from record.frames import Frame
from record.record import ConfigParser, ConfigurationError

from . import mock_read, TEST_FIXTURE_DIR


@pytest.fixture
def json_mock(mocker):
    return mocker.patch.object(
        json, 'dumps', side_effect=json.dumps, autospec=True
    )


# NOTE: All timestamps need to be > 3600 to avoid breaking the tests on
# Windows.

# current

def test_current(mocker, record):
    content = json.dumps({'project': 'foo', 'start': 4000, 'tags': ['A', 'B']})

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert record.current['project'] == 'foo'
    assert record.current['start'] == arrow.get(4000)
    assert record.current['tags'] == ['A', 'B']


def test_current_with_empty_file(mocker, record):
    mocker.patch('builtins.open', mocker.mock_open(read_data=""))
    mocker.patch('os.path.getsize', return_value=0)
    assert record.current == {}


def test_current_with_nonexistent_file(mocker, record):
    mocker.patch('builtins.open', side_effect=IOError)
    assert record.current == {}


def test_current_record_non_valid_json(mocker, record):
    content = "{'foo': bar}"

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    mocker.patch('os.path.getsize', return_value=len(content))
    with pytest.raises(RecordError):
        record.current


def test_current_with_given_state(config_dir, mocker):
    content = json.dumps({'project': 'foo', 'start': 4000})
    record = Record(current={'project': 'bar', 'start': 4000},
                    config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert record.current['project'] == 'bar'


def test_current_with_empty_given_state(config_dir, mocker):
    content = json.dumps({'project': 'foo', 'start': 4000})
    record = Record(current=[], config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert record.current == {}


def test_current_as_running_frame(record):
    """
    Ensures frame can be created without a stop date.
    Catches #417: editing task in progress throws an exception
    """
    record.start('foo', tags=['A', 'B'])

    cur = record.current
    frame = Frame(cur['start'], None, cur['project'], None, cur['tags'])

    assert frame.stop is None
    assert frame.project == 'foo'
    assert frame.tags == ['A', 'B']


# last_sync

def test_last_sync(mocker, record):
    now = arrow.get(4123)
    content = json.dumps(now.int_timestamp)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert record.last_sync == now


def test_last_sync_with_empty_file(mocker, record):
    mocker.patch('builtins.open', mocker.mock_open(read_data=""))
    mocker.patch('os.path.getsize', return_value=0)
    assert record.last_sync == arrow.get(0)


def test_last_sync_with_nonexistent_file(mocker, record):
    mocker.patch('builtins.open', side_effect=IOError)
    assert record.last_sync == arrow.get(0)


def test_last_sync_record_non_valid_json(mocker, record):
    content = "{'foo': bar}"

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    mocker.patch('os.path.getsize', return_value=len(content))
    with pytest.raises(RecordError):
        record.last_sync


def test_last_sync_with_given_state(config_dir, mocker):
    content = json.dumps(123)
    now = arrow.now()
    record = Record(last_sync=now, config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert record.last_sync == now


def test_last_sync_with_empty_given_state(config_dir, mocker):
    content = json.dumps(123)
    record = Record(last_sync=None, config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert record.last_sync == arrow.get(0)


# frames

def test_frames(mocker, record):
    content = json.dumps([[4000, 4010, 'foo', None, ['A', 'B', 'C']]])

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert len(record.frames) == 1
    assert record.frames[0].project == 'foo'
    assert record.frames[0].start == arrow.get(4000)
    assert record.frames[0].stop == arrow.get(4010)
    assert record.frames[0].tags == ['A', 'B', 'C']


def test_frames_without_tags(mocker, record):
    content = json.dumps([[4000, 4010, 'foo', None]])

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert len(record.frames) == 1
    assert record.frames[0].project == 'foo'
    assert record.frames[0].start == arrow.get(4000)
    assert record.frames[0].stop == arrow.get(4010)
    assert record.frames[0].tags == []


def test_frames_with_empty_file(mocker, record):
    mocker.patch('builtins.open', mocker.mock_open(read_data=""))
    mocker.patch('os.path.getsize', return_value=0)
    assert len(record.frames) == 0


def test_frames_with_nonexistent_file(mocker, record):
    mocker.patch('builtins.open', side_effect=IOError)
    assert len(record.frames) == 0


def test_frames_record_non_valid_json(mocker, record):
    content = "{'foo': bar}"

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    mocker.patch('os.path.getsize', return_value=len(content))
    with pytest.raises(RecordError):
        record.frames


def test_given_frames(config_dir, mocker):
    content = json.dumps([[4000, 4010, 'foo', None, ['A']]])
    record = Record(frames=[[4000, 4010, 'bar', None, ['A', 'B']]],
                    config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert len(record.frames) == 1
    assert record.frames[0].project == 'bar'
    assert record.frames[0].tags == ['A', 'B']


def test_frames_with_empty_given_state(config_dir, mocker):
    content = json.dumps([[0, 10, 'foo', None, ['A']]])
    record = Record(frames=[], config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    assert len(record.frames) == 0


# config

def test_empty_config_dir():
    record = Record()
    assert record._dir == get_app_dir('record')


def test_wrong_config(mocker, record):
    content = """
toto
    """
    mocker.patch.object(ConfigParser, 'read', mock_read(content))
    with pytest.raises(ConfigurationError):
        record.config


def test_empty_config(mocker, record):
    mocker.patch.object(ConfigParser, 'read', mock_read(''))
    assert len(record.config.sections()) == 0


# start

def test_start_new_project(record):
    record.start('foo', ['A', 'B'])

    assert record.current != {}
    assert record.is_started is True
    assert record.current.get('project') == 'foo'
    assert isinstance(record.current.get('start'), arrow.Arrow)
    assert record.current.get('tags') == ['A', 'B']


def test_start_new_project_without_tags(record):
    record.start('foo')

    assert record.current != {}
    assert record.is_started is True
    assert record.current.get('project') == 'foo'
    assert isinstance(record.current.get('start'), arrow.Arrow)
    assert record.current.get('tags') == []


def test_start_two_projects(record):
    record.start('foo')

    with pytest.raises(RecordError):
        record.start('bar')

    assert record.current != {}
    assert record.current['project'] == 'foo'
    assert record.is_started is True


def test_start_default_tags(mocker, record):
    content = """
[default_tags]
my project = A B
    """

    mocker.patch.object(ConfigParser, 'read', mock_read(content))
    record.start('my project')
    assert record.current['tags'] == ['A', 'B']


def test_start_default_tags_with_supplementary_input_tags(mocker, record):
    content = """
[default_tags]
my project = A B
    """

    mocker.patch.object(ConfigParser, 'read', mock_read(content))
    record.start('my project', tags=['C', 'D'])
    assert record.current['tags'] == ['C', 'D', 'A', 'B']


def test_start_nogap(record):

    record.start('foo')
    record.stop()
    record.start('bar', gap=False)

    assert record.frames[-1].stop == record.current['start']


def test_start_project_at(record):
    now = arrow.now()
    record.start('foo', start_at=now)
    record.stop()

    # Task can't start before the previous task ends
    with pytest.raises(RecordError):
        time_str = '1970-01-01T00:00'
        time_obj = arrow.get(time_str)
        record.start('foo', start_at=time_obj)

    # Task can't start in the future
    with pytest.raises(RecordError):
        time_str = '2999-12-31T23:59'
        time_obj = arrow.get(time_str)
        record.start('foo', start_at=time_obj)

    assert record.frames[-1].start == now


# stop

def test_stop_started_project(record):
    record.start('foo', tags=['A', 'B'])
    record.stop()

    assert record.current == {}
    assert record.is_started is False
    assert len(record.frames) == 1
    assert record.frames[0].project == 'foo'
    assert isinstance(record.frames[0].start, arrow.Arrow)
    assert isinstance(record.frames[0].stop, arrow.Arrow)
    assert record.frames[0].tags == ['A', 'B']


def test_stop_started_project_without_tags(record):
    record.start('foo')
    record.stop()

    assert record.current == {}
    assert record.is_started is False
    assert len(record.frames) == 1
    assert record.frames[0].project == 'foo'
    assert isinstance(record.frames[0].start, arrow.Arrow)
    assert isinstance(record.frames[0].stop, arrow.Arrow)
    assert record.frames[0].tags == []


def test_stop_no_project(record):
    with pytest.raises(RecordError):
        record.stop()


def test_stop_started_project_at(record):
    record.start('foo')
    now = arrow.now()

    # Task can't end before it starts
    with pytest.raises(RecordError):
        time_str = '1970-01-01T00:00'
        time_obj = arrow.get(time_str)
        record.stop(stop_at=time_obj)

    # Task can't end in the future
    with pytest.raises(RecordError):
        time_str = '2999-12-31T23:59'
        time_obj = arrow.get(time_str)
        record.stop(stop_at=time_obj)

    record.stop(stop_at=now)
    assert record.frames[-1].stop == now


# cancel

def test_cancel_started_project(record):
    record.start('foo')
    record.cancel()

    assert record.current == {}
    assert len(record.frames) == 0


def test_cancel_no_project(record):
    with pytest.raises(RecordError):
        record.cancel()


# save

def test_save_without_changes(mocker, record, json_mock):
    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert not json_mock.called


def test_save_current(mocker, record, json_mock):
    record.start('foo', ['A', 'B'])

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert json_mock.call_count == 1
    result = json_mock.call_args[0][0]
    assert result['project'] == 'foo'
    assert isinstance(result['start'], (int, float))
    assert result['tags'] == ['A', 'B']


def test_save_current_without_tags(mocker, record, json_mock):
    record.start('foo')

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert json_mock.call_count == 1
    result = json_mock.call_args[0][0]
    assert result['project'] == 'foo'
    assert isinstance(result['start'], (int, float))
    assert result['tags'] == []

    dump_args = json_mock.call_args[1]
    assert dump_args['ensure_ascii'] is False


def test_save_empty_current(config_dir, mocker, json_mock):
    record = Record(current={}, config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open())

    record.current = {'project': 'foo', 'start': 4000}
    record.save()

    assert json_mock.call_count == 1
    result = json_mock.call_args[0][0]
    assert result == {'project': 'foo', 'start': 4000, 'tags': []}

    record.current = {}
    record.save()

    assert json_mock.call_count == 2
    result = json_mock.call_args[0][0]
    assert result == {}


def test_save_frames_no_change(config_dir, mocker, json_mock):
    record = Record(frames=[[4000, 4010, 'foo', None]],
                    config_dir=config_dir)

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert not json_mock.called


def test_save_added_frame(config_dir, mocker, json_mock):
    record = Record(frames=[[4000, 4010, 'foo', None]], config_dir=config_dir)
    record.frames.add('bar', 4010, 4020, ['A'])

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert json_mock.call_count == 1
    result = json_mock.call_args[0][0]
    assert len(result) == 2
    assert result[0][2] == 'foo'
    assert result[0][4] == []
    assert result[1][2] == 'bar'
    assert result[1][4] == ['A']


def test_save_changed_frame(config_dir, mocker, json_mock):
    record = Record(frames=[[4000, 4010, 'foo', None, ['A']]],
                    config_dir=config_dir)
    record.frames[0] = ('bar', 4000, 4010, ['A', 'B'])

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert json_mock.call_count == 1
    result = json_mock.call_args[0][0]
    assert len(result) == 1
    assert result[0][2] == 'bar'
    assert result[0][4] == ['A', 'B']

    dump_args = json_mock.call_args[1]
    assert dump_args['ensure_ascii'] is False


def test_save_config_no_changes(mocker, record):
    mocker.patch('builtins.open', mocker.mock_open())
    write_mock = mocker.patch.object(ConfigParser, 'write')
    record.save()

    assert not write_mock.called


def test_save_config(mocker, record):
    mocker.patch('builtins.open', mocker.mock_open())
    write_mock = mocker.patch.object(ConfigParser, 'write')
    record.config = ConfigParser()
    record.save()

    assert write_mock.call_count == 1


def test_save_last_sync(mocker, record, json_mock):
    now = arrow.now()
    record.last_sync = now

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert json_mock.call_count == 1
    assert json_mock.call_args[0][0] == now.int_timestamp


def test_save_empty_last_sync(config_dir, mocker, json_mock):
    record = Record(last_sync=arrow.now(), config_dir=config_dir)
    record.last_sync = None

    mocker.patch('builtins.open', mocker.mock_open())
    record.save()

    assert json_mock.call_count == 1
    assert json_mock.call_args[0][0] == 0


def test_record_save_calls_safe_save(mocker, config_dir, record):
    frames_file = os.path.join(config_dir, 'frames')
    record.start('foo', tags=['A', 'B'])
    record.stop()

    save_mock = mocker.patch('record.record.safe_save')
    record.save()

    assert record._frames.changed
    assert save_mock.call_count == 1
    assert len(save_mock.call_args[0]) == 2
    assert save_mock.call_args[0][0] == frames_file


# push

def test_push_with_no_config(record):
    config = ConfigParser()
    record.config = config

    with pytest.raises(RecordError):
        record.push(arrow.now())


def test_push_with_no_url(record):
    config = ConfigParser()
    config.add_section('backend')
    config.set('backend', 'token', 'bar')
    record.config = config

    with pytest.raises(RecordError):
        record.push(arrow.now())


def test_push_with_no_token(record):
    config = ConfigParser()
    config.add_section('backend')
    config.set('backend', 'url', 'http://foo.com')
    record.config = config

    with pytest.raises(RecordError):
        record.push(arrow.now())


def test_push(mocker, record):
    config = ConfigParser()
    config.add_section('backend')
    config.set('backend', 'url', 'http://foo.com')
    config.set('backend', 'token', 'bar')

    record.frames.add('foo', 4001, 4002)
    record.frames.add('foo', 4003, 4004)

    record.last_sync = arrow.now()

    record.frames.add('bar', 4001, 4002, ['A', 'B'])
    record.frames.add('lol', 4001, 4002)

    last_pull = arrow.now()

    record.frames.add('foo', 4001, 4002)
    record.frames.add('bar', 4003, 4004)

    mocker.patch.object(record, '_get_remote_projects', return_value=[
        {'name': 'foo', 'id': '08288b71-4500-40dd-96b1-a995937a15fd'},
        {'name': 'bar', 'id': 'f0534272-65fa-4832-a49e-0eedf68b3a84'},
        {'name': 'lol', 'id': '7fdaf65e-66bd-4c01-b09e-74bdc8cbe552'},
    ])

    class Response:
        def __init__(self):
            self.status_code = 201

    mock_put = mocker.patch('requests.post', return_value=Response())
    mocker.patch.object(
        Record, 'config', new_callable=mocker.PropertyMock, return_value=config
    )
    record.push(last_pull)

    requests.post.assert_called_once_with(
        mocker.ANY,
        mocker.ANY,
        headers={
            'content-type': 'application/json',
            'Authorization': "Token " + config.get('backend', 'token')
        }
    )

    frames_sent = json.loads(mock_put.call_args[0][1])
    assert len(frames_sent) == 2

    assert frames_sent[0].get('project') == 'bar'
    assert frames_sent[0].get('tags') == ['A', 'B']

    assert frames_sent[1].get('project') == 'lol'
    assert frames_sent[1].get('tags') == []


# pull

def test_pull_with_no_config(record):
    config = ConfigParser()
    record.config = config

    with pytest.raises(ConfigurationError):
        record.pull()


def test_pull_with_no_url(record):
    config = ConfigParser()
    config.add_section('backend')
    config.set('backend', 'token', 'bar')
    record.config = config

    with pytest.raises(ConfigurationError):
        record.pull()


def test_pull_with_no_token(record):
    config = ConfigParser()
    config.add_section('backend')
    config.set('backend', 'url', 'http://foo.com')
    record.config = config

    with pytest.raises(ConfigurationError):
        record.pull()


def test_pull(mocker, record):
    config = ConfigParser()
    config.add_section('backend')
    config.set('backend', 'url', 'http://foo.com')
    config.set('backend', 'token', 'bar')

    record.last_sync = arrow.now()

    record.frames.add(
        'foo', 4001, 4002, ['A', 'B'], id='1c006c6e6cc14c80ab22b51c857c0b06'
    )

    mocker.patch.object(record, '_get_remote_projects', return_value=[
        {'name': 'foo', 'id': '08288b71-4500-40dd-96b1-a995937a15fd'},
        {'name': 'bar', 'id': 'f0534272-65fa-4832-a49e-0eedf68b3a84'},
    ])

    class Response:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return [
                {
                    'id': '1c006c6e-6cc1-4c80-ab22-b51c857c0b06',
                    'project': 'foo',
                    'begin_at': 4003,
                    'end_at': 4004,
                    'tags': ['A']
                },
                {
                    'id': 'c44aa815-4d77-4a58-bddd-1afa95562141',
                    'project': 'bar',
                    'begin_at': 4004,
                    'end_at': 4005,
                    'tags': []
                }
            ]

    mocker.patch('requests.get', return_value=Response())
    mocker.patch.object(
        Record, 'config', new_callable=mocker.PropertyMock, return_value=config
    )
    record.pull()

    requests.get.assert_called_once_with(
        mocker.ANY,
        params={'last_sync': record.last_sync},
        headers={
            'content-type': 'application/json',
            'Authorization': "Token " + config.get('backend', 'token')
        }
    )

    assert len(record.frames) == 2

    assert record.frames[0].id == '1c006c6e6cc14c80ab22b51c857c0b06'
    assert record.frames[0].project == 'foo'
    assert record.frames[0].start.int_timestamp == 4003
    assert record.frames[0].stop.int_timestamp == 4004
    assert record.frames[0].tags == ['A']

    assert record.frames[1].id == 'c44aa8154d774a58bddd1afa95562141'
    assert record.frames[1].project == 'bar'
    assert record.frames[1].start.int_timestamp == 4004
    assert record.frames[1].stop.int_timestamp == 4005
    assert record.frames[1].tags == []


# projects

def test_projects(record):
    for name in ('foo', 'bar', 'bar', 'bar', 'foo', 'lol'):
        record.frames.add(name, 4000, 4000)

    assert record.projects == ['bar', 'foo', 'lol']


def test_projects_no_frames(record):
    assert record.projects == []


# tags

def test_tags(record):
    samples = (
        ('foo', ('A', 'D')),
        ('bar', ('A', 'C')),
        ('foo', ('B', 'C')),
        ('lol', ()),
        ('bar', ('C'))
    )

    for name, tags in samples:
        record.frames.add(name, 4000, 4000, tags)

    assert record.tags == ['A', 'B', 'C', 'D']


def test_tags_no_frames(record):
    assert record.tags == []


# merge

@pytest.mark.datafiles(
    TEST_FIXTURE_DIR / 'frames-with-conflict',
    )
def test_merge_report(record, datafiles):
    # Get report
    record.frames.add('foo', 4000, 4015, id='1', updated_at=4015)
    record.frames.add('bar', 4020, 4045, id='2', updated_at=4045)

    conflicting, merging = record.merge_report(
        str(datafiles) + '/frames-with-conflict')

    assert len(conflicting) == 1
    assert len(merging) == 1

    assert conflicting[0].id == '2'
    assert merging[0].id == '3'


def test_report(record):
    record.start('foo', tags=['A', 'B'])
    record.stop()

    report = record.report(arrow.now(), arrow.now())
    assert 'time' in report
    assert 'timespan' in report
    assert 'from' in report['timespan']
    assert 'to' in report['timespan']
    assert len(report['projects']) == 1
    assert report['projects'][0]['name'] == 'foo'
    assert len(report['projects'][0]['tags']) == 2
    assert report['projects'][0]['tags'][0]['name'] == 'A'
    assert 'time' in report['projects'][0]['tags'][0]
    assert report['projects'][0]['tags'][1]['name'] == 'B'
    assert 'time' in report['projects'][0]['tags'][1]

    record.start('bar', tags=['C'])
    record.stop()

    report = record.report(arrow.now(), arrow.now())
    assert len(report['projects']) == 2
    assert report['projects'][0]['name'] == 'bar'
    assert report['projects'][1]['name'] == 'foo'
    assert len(report['projects'][0]['tags']) == 1
    assert report['projects'][0]['tags'][0]['name'] == 'C'

    report = record.report(
        arrow.now(), arrow.now(), projects=['foo'], tags=['B']
    )
    assert len(report['projects']) == 1
    assert report['projects'][0]['name'] == 'foo'
    assert len(report['projects'][0]['tags']) == 1
    assert report['projects'][0]['tags'][0]['name'] == 'B'

    record.start('baz', tags=['D'])
    record.stop()

    report = record.report(arrow.now(), arrow.now(), projects=["foo"])
    assert len(report['projects']) == 1

    report = record.report(arrow.now(), arrow.now(), ignore_projects=["bar"])
    assert len(report['projects']) == 2

    report = record.report(arrow.now(), arrow.now(), tags=["A"])
    assert len(report['projects']) == 1

    report = record.report(arrow.now(), arrow.now(), ignore_tags=["D"])
    assert len(report['projects']) == 2

    with pytest.raises(RecordError):
        record.report(
            arrow.now(), arrow.now(), projects=["foo"], ignore_projects=["foo"]
        )

    with pytest.raises(RecordError):
        record.report(arrow.now(), arrow.now(), tags=["A"], ignore_tags=["A"])


def test_report_current(mocker, config_dir):
    mocker.patch('arrow.utcnow', return_value=arrow.get(5000))

    record = Record(
        current={'project': 'foo', 'start': 4000},
        config_dir=config_dir
    )

    for _ in range(2):
        report = record.report(
            arrow.utcnow(), arrow.utcnow(), current=True, projects=['foo']
        )
    assert len(report['projects']) == 1
    assert report['projects'][0]['name'] == 'foo'
    assert report['projects'][0]['time'] == pytest.approx(1000)

    report = record.report(
        arrow.utcnow(), arrow.utcnow(), current=False, projects=['foo']
    )
    assert len(report['projects']) == 0

    report = record.report(
        arrow.utcnow(), arrow.utcnow(), projects=['foo']
    )
    assert len(report['projects']) == 0


@pytest.mark.parametrize(
    "date_as_unixtime,include_partial,sum_", (
        (3600 * 24, False, 0.0),
        (3600 * 48, False, 0.0),
        (3600 * 24, True, 7200.0),
        (3600 * 48, True, 3600.0),
    )
)
def test_report_include_partial_frames(mocker, record, date_as_unixtime,
                                       include_partial, sum_):
    """Test report building with frames that cross report boundaries

    1 event is added that has 2 hours in one day and 1 in the next. The
    parametrization checks that the report for both days is empty with
    `include_partial=False` and report the correct amount of hours with
    `include_partial=False`

    """
    content = json.dumps([[
        3600 * 46,
        3600 * 49,
        "programming",
        "3e76c820909840f89cabaf106ab7d12a",
        ["cli"],
        1548797432
    ]])
    mocker.patch('builtins.open', mocker.mock_open(read_data=content))
    date = arrow.get(date_as_unixtime)
    report = record.report(
        from_=date, to=date, include_partial_frames=include_partial,
    )
    assert report["time"] == pytest.approx(sum_, abs=1e-3)


# renaming project updates frame last_updated time
def test_rename_project_with_time(record):
    """
    Renaming a project should update the "last_updated" time on any frame that
    contains that project.
    """
    record.frames.add(
        'foo', 4001, 4002, ['some_tag'],
        id='c76d1ad0282c429595cc566d7098c165', updated_at=4005
    )
    record.frames.add(
        'bar', 4010, 4015, ['other_tag'],
        id='eed598ff363d42658a095ae6c3ae1088', updated_at=4035
    )

    record.rename_project("foo", "baz")

    assert len(record.frames) == 2

    assert record.frames[0].id == 'c76d1ad0282c429595cc566d7098c165'
    assert record.frames[0].project == 'baz'
    assert record.frames[0].start.int_timestamp == 4001
    assert record.frames[0].stop.int_timestamp == 4002
    assert record.frames[0].tags == ['some_tag']
    # assert record.frames[0].updated_at.int_timestamp == 9000
    assert record.frames[0].updated_at.int_timestamp > 4005

    assert record.frames[1].id == 'eed598ff363d42658a095ae6c3ae1088'
    assert record.frames[1].project == 'bar'
    assert record.frames[1].start.int_timestamp == 4010
    assert record.frames[1].stop.int_timestamp == 4015
    assert record.frames[1].tags == ['other_tag']
    assert record.frames[1].updated_at.int_timestamp == 4035


def test_rename_tag_with_time(record):
    """
    Renaming a tag should update the "last_updated" time on any frame that
    contains that tag.
    """
    record.frames.add(
        'foo', 4001, 4002, ['some_tag'],
        id='c76d1ad0282c429595cc566d7098c165', updated_at=4005
    )
    record.frames.add(
        'bar', 4010, 4015, ['other_tag'],
        id='eed598ff363d42658a095ae6c3ae1088', updated_at=4035
    )

    record.rename_tag("other_tag", "baz")

    assert len(record.frames) == 2

    assert record.frames[0].id == 'c76d1ad0282c429595cc566d7098c165'
    assert record.frames[0].project == 'foo'
    assert record.frames[0].start.int_timestamp == 4001
    assert record.frames[0].stop.int_timestamp == 4002
    assert record.frames[0].tags == ['some_tag']
    assert record.frames[0].updated_at.int_timestamp == 4005

    assert record.frames[1].id == 'eed598ff363d42658a095ae6c3ae1088'
    assert record.frames[1].project == 'bar'
    assert record.frames[1].start.int_timestamp == 4010
    assert record.frames[1].stop.int_timestamp == 4015
    assert record.frames[1].tags == ['baz']
    # assert record.frames[1].updated_at.int_timestamp == 9000
    assert record.frames[1].updated_at.int_timestamp > 4035

# add


def test_add_success(record):
    """
    Adding a new frame outside of live tracking successfully
    """
    record.add(project="test_project", tags=['fuu', 'bar'],
               from_date=6000, to_date=7000)

    assert len(record.frames) == 1
    assert record.frames[0].project == "test_project"
    assert 'fuu' in record.frames[0].tags


def test_add_failure(record):
    """
    Adding a new frame outside of live tracking fails when
    to date is before from date
    """
    with pytest.raises(RecordError):
        record.add(project="test_project", tags=['fuu', 'bar'],
                   from_date=7000, to_date=6000)


def test_validate_report_options(record):
    assert record._validate_report_options(["project_foo"], None)
    assert record._validate_report_options(None, ["project_foo"])
    assert not record._validate_report_options(["project_foo"],
                                               ["project_foo"])
    assert record._validate_report_options(["project_foo"], ["project_bar"])
    assert not record._validate_report_options(["project_foo", "project_bar"],
                                               ["project_foo"])
    assert not record._validate_report_options(["project_foo", "project_bar"],
                                               ["project_foo", "project_bar"])
    assert record._validate_report_options(None, None)
