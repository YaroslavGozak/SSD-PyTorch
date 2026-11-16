import tempfile
import unittest
from pathlib import Path

from experiments.roi_merge_b.common import atomic,sha
from experiments.roi_merge_b.execution import Store
from experiments.roi_merge_b.stages import load_calibration


class RecoveryTests(unittest.TestCase):
    def test_atomic_resume_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp,{'manifest':'a','threads':4})
            store.put('trials',[32,64],{'trials':[1,2]})
            restored=Store(tmp,{'manifest':'a','threads':4})
            self.assertEqual(restored.get('trials',[32,64]),{'trials':[1,2]})
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                Store(tmp,{'manifest':'a','threads':3})

    def test_cost_receipt_and_heldout_artifact_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cost_model.json'
            atomic(path,{'role':'calibration','calibration_id':'a'})
            atomic(Path(tmp)/'cost_model.receipt.json',{'sha256':sha(path)})
            self.assertEqual(load_calibration(tmp)['role'],'calibration')
            atomic(path,{'role':'heldout_reference'})
            with self.assertRaisesRegex(ValueError,'receipt mismatch'):
                load_calibration(tmp)
            atomic(Path(tmp)/'cost_model.receipt.json',{'sha256':sha(path)})
            with self.assertRaisesRegex(ValueError,'Not a deployable'):
                load_calibration(tmp)


if __name__=='__main__':
    unittest.main()
