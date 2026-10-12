"""User-authorized Taiwan proxy restart; scientific configuration unchanged."""
import contextlib,io,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlsplit
from experiments import resume_controller_research as restart
from experiments import controller_research as base

ROOT=Path('results/controller_tw_20260929')
AUTH='controller_tw_authorization_20260929'
DEADLINE=datetime(2026,9,29,3,32,29,tzinfo=timezone.utc).timestamp()

def configure():
    restart.ROOT=ROOT;restart.AUTH=AUTH;restart.DEADLINE=DEADLINE;restart.configure()

def run():
    configure()
    proxy=urlsplit(os.environ.get('HTTPS_PROXY',''))
    if proxy.hostname!='127.0.0.1' or proxy.port!=17890:raise RuntimeError('authorized local proxy required')
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):base.advance()
    lines=capture.getvalue().splitlines()
    if lines:print(lines[-1],flush=True)

def prepare():
    configure();restart.prepare()
    from experiments.puzzle_pilot import save
    save(ROOT/'transport.json',{'route':'authorized_TW_proxy','previous_attempts':[
        'results/controller_research_20260928/dev/react/cr-dev-rush-0/result.json',
        'results/controller_resume_20260929/dev/react/cr-dev-rush-0/result.json'],
        'note':'Explicit second infrastructure retry for first dev ReAct episode; all other tasks unattempted.'})

if __name__=='__main__':
    configure()
    {'prepare':prepare,'evaluate':base.evaluate,'audit':restart.audit}[sys.argv[1]]()
