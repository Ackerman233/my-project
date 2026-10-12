from dynamic_skill.space_boundary import discovery_check,boundary_prompt,target_type
from dynamic_skill.control_only_evolution import ControlExecutor,ControlPolicy

def test_prompt_preserves_protocol_no_forced_minimum():
    p=boundary_prompt('Stop before unknown information. Return thought/actions.')
    assert 'Stop before unknown information.' not in p
    assert 'one action is correct' in p and 'thought/actions' in p

def test_discovery_requires_exact_target_and_current_evidence():
    args=('put some egg on microwave.','go to table 1',['go to table 2'])
    assert discovery_check(*args,['take egg 2 from table 1'],0)['triggered']
    assert not discovery_check(*args,['take mug 2 from table 1'],0)['triggered']
    assert not discovery_check(*args,[],0)['triggered']
    assert target_type('unsupported task') is None

def test_no_cut_acquisition_tail_or_empty_queue():
    for tail in ([],['take egg 2 from table 1'],['go to table 1']):
        assert not discovery_check('put some egg on microwave.','go to table 1',tail,['take egg 2 from table 1'],0)['triggered']

def test_real_cut_only_when_enabled_not_failure():
    for enabled in (False,True):
        c=ControlExecutor(ControlPolicy(),discovery_interrupt=enabled)
        c.start('Room','put some egg on microwave.')
        c.load_chunk(['go to table 1','go to table 2'])
        a=c.next_action();c.before_action(a,'Room',[a]);c.after_action(['take egg 2 from table 1'])
        c.observe(a,'On table 1 you see egg 2.')
        assert c.interruption_count==int(enabled)
        assert not c.failures
        assert c.last_execution_audit['discovery_pause']==enabled
