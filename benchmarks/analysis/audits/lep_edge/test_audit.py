"""Small deterministic tests of the source-specific extraction and normalization."""
import math
from pathlib import Path
import unittest

from audit import COLOURS,normalized_density,recover_operations,run


def source_operations():
    operations=[([100,100,400,400],b"re")]
    for colour in COLOURS:
        operations += [(list(colour),b"RG"),([100,100],b"m")]
        for index in range(100):
            height=100+4*(1+index%10)
            operations += [([100+4*index,height],b"l"),([104+4*index,height],b"l")]
        operations.append(([],b"S"))
    return operations


class AuditTests(unittest.TestCase):
    def test_local_coordinates_and_series(self):
        rows,frame=recover_operations(source_operations())
        self.assertEqual(frame,[100.,100.,400.,400.])
        self.assertEqual(len(rows),300)
        self.assertEqual(rows[0]["theta_lo_urad"],-20)
        self.assertEqual(rows[99]["theta_hi_urad"],30)
        self.assertEqual(rows[0]["ordinate_au"],1000)
        self.assertEqual(rows[100]["series"],"early")

    def test_unexpected_frame_and_colour_rejected(self):
        with self.assertRaisesRegex(ValueError,"unique plotting frame"):
            recover_operations(source_operations()[1:])
        operations=source_operations(); operations[1]=([0,1,0],b"RG")
        with self.assertRaisesRegex(ValueError,"stroke colour"):
            recover_operations(operations)

    def test_density_preserves_area_for_unequal_bins(self):
        rows=[dict(ordinate_au=2,theta_lo_urad=0,theta_hi_urad=1),
              dict(ordinate_au=1,theta_lo_urad=1,theta_hi_urad=3)]
        self.assertEqual(normalized_density(rows),[.5,.25])
        rows[0]["ordinate_au"]=math.nan
        with self.assertRaisesRegex(ValueError,"ordinates"):
            normalized_density(rows)

    def test_wrong_source_fails_before_pdf_parse_or_output(self):
        with self.assertRaisesRegex(ValueError,"Source PDF hash"):
            run(Path(__file__),Path("UNREAD_REFERENCE"),Path("UNREAD_REDUCTION"),Path("UNCREATED_OUTPUT"))


if __name__=="__main__":
    unittest.main()
