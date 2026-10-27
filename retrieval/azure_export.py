"""Optional private Blob Storage copies of completed dataset snapshots.

No source upload or cloud processing: the ZIP is the exact local artifact.
Credentials remain in the backend's Azure Identity credential chain.
"""
import importlib.util
import json
import os
import re
import time
import zipfile
from contextlib import contextmanager

from . import dataset, storage as db, windows


class ExportError(ValueError):
    """Safe, application-authored message suitable for returning to the UI."""


def configuration():
    account = os.environ.get('TOUCHLINE_AZURE_ACCOUNT_URL', '').strip().rstrip('/')
    container = os.environ.get('TOUCHLINE_AZURE_CONTAINER', '').strip()
    if not account or not container:
        raise ExportError('Azure is not configured. Follow the Azure export setup in the README.')
    if not re.fullmatch(r'https://[a-z0-9]{3,24}\.blob\.core\.windows\.net', account):
        raise ExportError('Set TOUCHLINE_AZURE_ACCOUNT_URL to an HTTPS Azure Blob account URL, without a token or path.')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]', container) or '--' in container:
        raise ExportError('Set TOUCHLINE_AZURE_CONTAINER to a valid private container name.')
    return account, container


def sdk_available():
    try:
        return all(importlib.util.find_spec(name) is not None for name in ('azure.identity', 'azure.storage.blob'))
    except ModuleNotFoundError:
        return False


def settings():
    try:
        account, container = configuration()
    except ValueError as error:
        return {'configured': False, 'available': False, 'message': str(error)}
    available = sdk_available()
    return {'configured': True, 'available': available, 'account_url': account, 'container': container,
            'message': 'Configured; access is checked when saving or downloading.' if available else
            'Install Azure support with: python -m pip install -e ".[azure]"'}


@contextmanager
def _container(account, name):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient
    with DefaultAzureCredential(exclude_interactive_browser_credential=True) as credential:
        with ContainerClient(account, name, credential=credential,
                             connection_timeout=10, read_timeout=60, retry_total=2) as client:
            # Refuse accidental public publication, even if an existing container
            # was configured by somebody else. This app never changes its ACL.
            if client.get_container_properties().get('public_access'):
                raise ExportError('Choose a private Azure container. Anonymous public access must be disabled.')
            yield client


def history(jobid):
    db.get('jobs', jobid)
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT data FROM events WHERE job=? AND kind='azure_export_saved' ORDER BY id DESC", (jobid,))
        return [json.loads(row['data']) for row in rows]


def status(jobid):
    job = db.get('jobs', jobid)
    config = settings()
    signature = windows.signature(jobid)
    records = history(jobid)
    for record in records:
        record['historical'] = record['snapshot_signature'] != signature
        record['download_available'] = bool(config['available'] and
            config['account_url'] == record['account_url'] and config['container'] == record['container'])
        record['download_url'] = f'/api/jobs/{jobid}/azure-exports/{record["id"]}/download'
    return config | {'status': job.get('azure_status', 'idle'), 'error': job.get('azure_error'),
                     'progress': job.get('azure_progress', 0), 'total': job.get('azure_total', 0),
                     'exports': records}


def save(jobid):
    """Upload a fresh ready ZIP once per content hash; never overwrite blobs."""
    try:
        account, name = configuration()
        if not sdk_available():
            raise ExportError('Install Azure support with: python -m pip install -e ".[azure]"')
        try:
            path = dataset.ready_path(jobid)
        except ValueError as error:
            raise ExportError(str(error)) from None
        digest = db.sha_file(path)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
        # The job may have changed while hashing or while this task was queued.
        try:
            current_path = dataset.ready_path(jobid)
        except ValueError as error:
            raise ExportError(str(error)) from None
        if current_path != path:
            raise ExportError('The prepared dataset changed. Save the new export instead.')
        size = path.stat().st_size
        blob_name = f'touchline/{jobid}/{digest}/dataset.zip'
        record = {'id': digest, 'created': time.time(), 'account_url': account, 'container': name,
                  'blob': blob_name, 'sha256': digest, 'bytes': size,
                  'snapshot_signature': manifest['snapshot_signature'], 'mode': manifest['mode'],
                  'tracklet_count': manifest['tracklet_count'], 'window_count': manifest['window_count'],
                  'included_video': manifest['included_video']}
        db.update_job(jobid, azure_status='working', azure_error=None, azure_progress=0, azure_total=size)
        from azure.core.exceptions import ResourceExistsError
        from azure.storage.blob import ContentSettings
        with _container(account, name) as client:
            blob = client.get_blob_client(blob_name)
            try:
                with path.open('rb') as data:
                    blob.upload_blob(data, overwrite=False, length=size,
                                     content_settings=ContentSettings(content_type='application/zip'),
                                     metadata={'sha256': digest},
                                     progress_hook=lambda sent, total: db.update_job(jobid, azure_progress=sent))
            except ResourceExistsError:
                # Retry after a timeout/restart: verify the existing snapshot.
                pass
            properties = blob.get_blob_properties()
            if properties.size != size or properties.metadata.get('sha256') != digest:
                raise ExportError('The Azure copy does not match this dataset. No existing blob was overwritten.')
        previous = history(jobid)
        if not any(r['id'] == digest and r['account_url'] == account and r['container'] == name for r in previous):
            db.event(jobid, 'azure_export_saved', record)
        db.update_job(jobid, azure_status='ready', azure_error=None, azure_progress=size)
        return record
    except Exception as error:
        # SDK exceptions may contain URLs, request details or credential errors.
        # Keep those out of job JSON, API responses and frontend notifications.
        message = str(error) if isinstance(error, ExportError) else (
            'Azure save failed. Check your sign-in, private container, network and Storage Blob Data Contributor role, then retry.')
        db.update_job(jobid, azure_status='error', azure_error=message)
        raise ExportError(message) from None


def download(jobid, exportid):
    """Stream a saved snapshot even after its local ZIP has been removed."""
    record = next((row for row in history(jobid) if row['id'] == exportid and
                   (row['account_url'], row['container']) == configuration()), None)
    if record is None:
        raise KeyError(exportid)
    if not sdk_available():
        raise ExportError('Install Azure support with: python -m pip install -e ".[azure]"')
    manager = _container(record['account_url'], record['container'])
    try:
        client = manager.__enter__()
        blob = client.get_blob_client(record['blob'])
        # Pin subsequent range reads to this version, including after a retry.
        properties = blob.get_blob_properties()
        if properties.size != record['bytes'] or properties.metadata.get('sha256') != record['sha256']:
            raise ExportError('The saved Azure snapshot has changed. Download cancelled.')
        from azure.core import MatchConditions
        stream = blob.download_blob(etag=properties.etag, match_condition=MatchConditions.IfNotModified)
    except Exception as error:
        manager.__exit__(type(error), error, error.__traceback__)
        message = str(error) if isinstance(error, ExportError) else (
            'Azure download failed. Check your sign-in, container, network and saved blob.')
        raise ExportError(message) from None

    def chunks():
        try:
            yield from stream.chunks()
        finally:
            manager.__exit__(None, None, None)
    return chunks(), record
