from experiments.audit_equal_operator_attribution import actual_tail_interruption


def test_terminal_cleanup_is_not_a_real_mid_queue_interruption():
    row={'done':True,'interruption_reason':None,'actual_discarded_actions':['go to desk 1']}
    assert not actual_tail_interruption(row)
    row['done']=False
    assert not actual_tail_interruption(row)
    row['interruption_reason']='Acquisition contradiction'
    assert actual_tail_interruption(row)
    row['actual_discarded_actions']=[]
    assert not actual_tail_interruption(row)
