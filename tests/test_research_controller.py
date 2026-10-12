import unittest
from dynamic_skill.research_controller import Controller, POLICIES


class ControllerTests(unittest.TestCase):
    def test_fixed_retains_tail(self):
        c = Controller(POLICIES['fixed'])
        self.assertFalse(c.observe('s','A+1',False,'s',['B+1'])['interrupt'])

    def test_invalid_interrupt_and_recovery(self):
        c = Controller(POLICIES['adaptive3'])
        self.assertEqual(c.action_limit(),4)
        self.assertTrue(c.observe('s','bad',False,'s',['x'])['interrupt'])
        self.assertEqual(c.action_limit(),1)
        for i in range(3): c.observe(str(i),'ok',True,str(i+1),[])
        self.assertEqual(c.action_limit(),4)

    def test_memory_is_state_specific(self):
        c = Controller(POLICIES['memory'])
        c.observe('s','bad',False,'s',[])
        self.assertEqual(c.memory('s'),['bad'])
        self.assertEqual(c.memory('different'),[])

    def test_patient_threshold(self):
        c = Controller(POLICIES['patient_memory'])
        self.assertFalse(c.observe('s','bad1',False,'s',['bad2'])['interrupt'])
        self.assertTrue(c.observe('s','bad2',False,'s',['ok'])['interrupt'])

    def test_react_one_action(self):
        self.assertEqual(Controller(POLICIES['react']).action_limit(),1)

if __name__ == '__main__': unittest.main()
