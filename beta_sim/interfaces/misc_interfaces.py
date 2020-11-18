import os
from shutil import copy
from glob import glob

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    SimpleInterface, traits
    )


class CopyEventsInputSpec(BaseInterfaceInputSpec):
    event_files = traits.List(item_trait=traits.File(exists=True))
    out_dir = traits.Directory()


class CopyEventsOutputSpec(TraitedSpec):
    moved_event_files = traits.List(item_trait=traits.File(exists=True))


class CopyEvents(SimpleInterface):
    input_spec = CopyEventsInputSpec
    output_spec = CopyEventsOutputSpec

    def _run_interface(self, runtime):
        out = os.path.join(self.inputs.out_dir, "eventfiles")
        os.makedirs(out, exist_ok=True)
        for e_file in self.inputs.event_files:
            copy(e_file, out)

        self._results['moved_event_files'] = glob(os.path.join(out, '*.tsv'))

        return runtime
