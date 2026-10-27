"""Cloud boundary tests use isolated data and an in-memory Blob client, no account."""
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from retrieval import azure_export as azure, dataset, storage as db, windows
from test_dataset import collection, decide

_original_container = azure._container


@pytest.fixture
def cloud(collection, monkeypatch):
    # The Azure SDK is optional. Mock only its boundary types when not installed;
    # when installed, these tests exercise the real ContentSettings/exception API.
    try:
        from azure.core.exceptions import ResourceExistsError
    except ImportError:
        import sys
        from types import ModuleType
        modules = {name: ModuleType(name) for name in
                   ('azure', 'azure.core', 'azure.core.exceptions', 'azure.storage', 'azure.storage.blob')}
        class ResourceExistsError(Exception):
            pass
        modules['azure.core.exceptions'].ResourceExistsError = ResourceExistsError
        modules['azure.core'].MatchConditions = SimpleNamespace(IfNotModified='unchanged')
        modules['azure.storage.blob'].ContentSettings = lambda **kw: SimpleNamespace(**kw)
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('TOUCHLINE_AZURE_ACCOUNT_URL', 'https://touchlinetest.blob.core.windows.net')
    monkeypatch.setenv('TOUCHLINE_AZURE_CONTAINER', 'datasets')
    monkeypatch.setattr(azure, 'sdk_available', lambda: True)
    blobs, state = {}, {'public': False, 'closed': 0, 'fail': False}

    class Blob:
        def __init__(self, name):
            self.name = name

        def upload_blob(self, data, *, overwrite, length, content_settings, metadata, progress_hook):
            assert not overwrite
            assert content_settings.content_type == 'application/zip'
            if state['fail']:
                raise ValueError('secret-token-and-account-key-do-not-leak')
            if self.name in blobs:
                raise ResourceExistsError('exists')
            content = data.read()
            assert length == len(content)
            blobs[self.name] = {'content': content, 'metadata': metadata}
            progress_hook(length, length)

        def get_blob_properties(self):
            value = blobs[self.name]
            return SimpleNamespace(size=len(value['content']), metadata=value['metadata'], etag='version-1')

        def download_blob(self, **kw):
            assert kw['etag'] == 'version-1'
            return SimpleNamespace(chunks=lambda: iter([blobs[self.name]['content']]))

    @contextmanager
    def container(account, name):
        try:
            if state['public']:
                raise azure.ExportError('Choose a private Azure container. Anonymous public access must be disabled.')
            yield SimpleNamespace(get_blob_client=Blob)
        finally:
            state['closed'] += 1

    monkeypatch.setattr(azure, '_container', container)
    return blobs, state


def prepare():
    windows.build('job')
    return dataset.export('job')


def test_unconfigured_azure_does_not_break_local_download(collection, monkeypatch):
    monkeypatch.delenv('TOUCHLINE_AZURE_ACCOUNT_URL', raising=False)
    monkeypatch.delenv('TOUCHLINE_AZURE_CONTAINER', raising=False)
    prepare()
    from retrieval.api import app
    with TestClient(app) as client:
        state = client.get('/api/jobs/job/azure-exports').json()
        assert state['configured'] is False and state['exports'] == []
        assert client.post('/api/jobs/job/azure-exports').status_code == 400
        assert client.get('/api/jobs/job/dataset/download').status_code == 200


def test_private_save_exact_bytes_retry_and_download_after_local_deletion(cloud):
    blobs, state = cloud
    path = prepare()
    expected = path.read_bytes()
    record = azure.save('job')
    assert record['sha256'] == hashlib.sha256(expected).hexdigest()
    assert blobs[record['blob']]['content'] == expected
    azure.save('job')  # ResourceExistsError on retry is verified and deduplicated.
    assert len(blobs) == len(azure.history('job')) == 1
    assert azure.status('job')['status'] == 'ready'
    path.unlink()
    from retrieval.api import app
    with TestClient(app) as client:
        saved = client.get('/api/jobs/job/azure-exports').json()['exports'][0]
        assert saved['download_available'] and not saved['historical']
        response = client.get(saved['download_url'])
        assert response.status_code == 200 and response.content == expected
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['content-length'] == str(len(expected))
        assert client.get('/api/jobs/job/azure-exports/foreign/download').status_code == 404
    assert state['closed'] == 3


def test_stale_zip_cannot_be_uploaded_but_saved_history_remains(cloud):
    blobs, _ = cloud
    prepare()
    record = azure.save('job')
    decide('accepted', 1, 4)
    with pytest.raises(ValueError, match='stale'):
        azure.save('job')
    assert len(blobs) == 1
    assert azure.status('job')['exports'][0]['historical']
    chunks, _ = azure.download('job', record['id'])
    assert b''.join(chunks) == blobs[record['blob']]['content']


def test_failures_never_publish_success_or_leak_sdk_details(cloud):
    blobs, state = cloud
    prepare()
    state['public'] = True
    with pytest.raises(ValueError, match='private'):
        azure.save('job')
    assert not blobs and azure.history('job') == []
    state.update(public=False, fail=True)
    with pytest.raises(ValueError, match='Azure save failed'):
        azure.save('job')
    saved = json.dumps(db.get('jobs', 'job'))
    assert 'secret-token' not in saved
    assert azure.status('job')['status'] == 'error'
    assert not blobs and azure.history('job') == []
    assert dataset.ready_path('job').is_file()
    state['fail'] = False
    azure.save('job')
    assert azure.status('job')['status'] == 'ready'


def test_changed_blob_is_not_overwritten_or_downloaded(cloud):
    blobs, _ = cloud
    prepare()
    record = azure.save('job')
    blobs[record['blob']]['metadata'] = {'sha256': 'changed'}
    with pytest.raises(ValueError, match='does not match'):
        azure.save('job')
    with pytest.raises(ValueError, match='has changed'):
        azure.download('job', record['id'])
    assert blobs[record['blob']]['metadata']['sha256'] == 'changed'


def test_background_api_and_restart_recovery(cloud):
    prepare()
    from retrieval.api import app, tasks
    db.update_job('job', azure_status='working')
    with TestClient(app) as client:
        assert client.get('/api/jobs/job/azure-exports').json()['status'] == 'error'
        assert client.post('/api/jobs/job/azure-exports').status_code == 200
        tasks['azure-job'].result(timeout=10)
        state = client.get('/api/jobs/job/azure-exports').json()
        assert state['status'] == 'ready' and len(state['exports']) == 1


def test_configuration_rejects_token_urls_and_invalid_container(cloud, monkeypatch):
    monkeypatch.setenv('TOUCHLINE_AZURE_ACCOUNT_URL', 'https://touchlinetest.blob.core.windows.net?sig=secret')
    assert not azure.settings()['available']
    assert 'secret' not in json.dumps(azure.settings())
    monkeypatch.setenv('TOUCHLINE_AZURE_ACCOUNT_URL', 'https://touchlinetest.blob.core.windows.net')
    monkeypatch.setenv('TOUCHLINE_AZURE_CONTAINER', 'not--private')
    assert not azure.settings()['available']


def test_real_container_boundary_checks_acl_and_closes_clients(cloud, monkeypatch):
    import importlib
    import sys
    from types import ModuleType
    try:
        identity = importlib.import_module('azure.identity')
    except ImportError:
        identity = ModuleType('azure.identity')
        monkeypatch.setitem(sys.modules, 'azure.identity', identity)
    storage_sdk = importlib.import_module('azure.storage.blob')
    state = {'access': None, 'closed': [], 'read': False}

    class Credential:
        def __init__(self, **kw):
            assert kw['exclude_interactive_browser_credential'] is True
        def __enter__(self):
            return self
        def __exit__(self, *args):
            state['closed'].append('credential')

    class Client:
        def __init__(self, account, name, **kw):
            assert isinstance(kw['credential'], Credential)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            state['closed'].append('client')
        def get_container_properties(self):
            state['read'] = True
            return {'public_access': state['access']}

    monkeypatch.setattr(identity, 'DefaultAzureCredential', Credential, raising=False)
    monkeypatch.setattr(storage_sdk, 'ContainerClient', Client, raising=False)
    # Restore the genuine boundary, rather than the cloud fixture's mock.
    with _original_container(*azure.configuration()):
        assert state['read']
    assert state['closed'] == ['client', 'credential']
    state['access'] = 'blob'
    with pytest.raises(ValueError, match='private'):
        with _original_container(*azure.configuration()):
            pytest.fail('Public container must be rejected before upload')
    assert state['closed'] == ['client', 'credential', 'client', 'credential']
