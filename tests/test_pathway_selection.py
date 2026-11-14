import unittest

from pathsyn.omics import select_pathways


class TestPathwaySelection(unittest.TestCase):
    def test_top_variance_and_gene_size_filtering(self):
        collections = {
            "HALLMARK": {
                "H1": [f"G{i}" for i in range(10)],
                "H_TOO_SMALL": ["G1", "G2"],
            },
            "KEGG": {
                "K_LOW": [f"G{i}" for i in range(10)],
                "K_HIGH": [f"G{i}" for i in range(11, 21)],
                "K_TOO_LARGE": [f"G{i}" for i in range(1001)],
            },
            "REACTOME": {
                "R_LOW": [f"G{i}" for i in range(10)],
                "R_HIGH": [f"G{i}" for i in range(11, 21)],
            },
        }
        universe = {f"G{i}" for i in range(1001)}
        scores = {
            "K_LOW": 0.1,
            "K_HIGH": 0.9,
            "R_LOW": 0.2,
            "R_HIGH": 0.8,
        }

        selected = select_pathways(
            collections,
            universe,
            scores,
            min_size=10,
            max_size=500,
            n_kegg=1,
            n_reactome=1,
        )

        self.assertIn("H1", selected)
        self.assertNotIn("H_TOO_SMALL", selected)
        self.assertIn("K_HIGH", selected)
        self.assertNotIn("K_LOW", selected)
        self.assertNotIn("K_TOO_LARGE", selected)
        self.assertIn("R_HIGH", selected)
        self.assertNotIn("R_LOW", selected)


if __name__ == "__main__":
    unittest.main()
