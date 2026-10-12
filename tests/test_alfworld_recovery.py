from dynamic_skill.alfworld_recovery import recovery_prompt

def call(trace,anomaly):
    return recovery_prompt('BASE',task='task',initial='visible room',observation='now',
                           admissible=['look'],trace=trace,anomaly=anomaly)

def test_inactive_identical():
    assert call([], '')=='BASE'

def test_failed_action_and_history_are_grounded():
    trace=[{'action':'go to desk 1','valid':True,'env_feedback':'You arrive at desk 1. You see bowl 1.'},
           {'action':'take bowl 1','valid':False,'env_feedback':'Nothing happens.'}]
    p=call(trace,'invalid action')
    assert 'take bowl 1' in p and 'Nothing happens.' in p and 'bowl 1' in p
    assert 'historical_location_observations' in p and 'not current-state guarantees' in p
    assert 'oracle' not in p and 'desklamp' not in p

def test_evidence_bounded():
    trace=[{'action':f'bad {i}','valid':False,'env_feedback':'Nothing happens.'} for i in range(40)]
    p=call(trace,'invalid');assert 'bad 39' in p and 'bad 0"' not in p
