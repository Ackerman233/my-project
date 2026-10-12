import pytest
from dynamic_skill.micro_loop_protocol import *

def test_controller():
    assert parse_controller(controller_block(INITIAL_CONTROL))==INITIAL_CONTROL
    with pytest.raises(ValueError):parse_controller(controller_block({**INITIAL_CONTROL,'max_actions':99}))

def test_thought_required():
    assert parse_response('<thought>Find the object.</thought><actions>["look"]</actions>',1)==(['look'],'Find the object.')
    with pytest.raises(ValueError):parse_response('<actions>["look"]</actions>',1)
    with pytest.raises(ValueError):parse_response('<thought>Go.</thought><actions>["look","look"]</actions>',1)

def test_same_inputs_same_prompt():
    f=make_prompt('skill',1)
    assert 'Thought/Action/Observation' in f('',task='t',initial='i',observation='o',admissible=['look'],trace=[],anomaly='')
    assert SkillChecker('invalid').check('look','Nothing happens.').should_interrupt
    assert not SkillChecker('invalid').check('look','same',['look','look'],['same','same']).should_interrupt
