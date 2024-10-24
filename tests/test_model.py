import numpy as np
from alfabet_lite import model


def test_predict():
    results = model.predict(["CC", "NCCO", "CF", "B"])

    assert not results[results.molecule == "B"].is_valid.any()
    assert results[results.molecule != "B"].is_valid.all()

    # Should be less than 1 kcal/mol on this easy set
    assert (results.bde_pred - results.bde).abs().mean() < 1.0

    np.testing.assert_allclose(
        results[results.molecule == "CC"].bde_pred, [90.7, 99.8], atol=1.0, rtol=0.05
    )

    np.testing.assert_allclose(
        results[results.molecule == "NCCO"].bde_pred,
        [90.0, 82.1, 98.2, 99.3, 92.1, 92.5, 105.2],
        atol=1.0,
        rtol=0.05,
    )


def test_data_missing():
    results = model.predict(["CCCCCOC"])
    assert np.isfinite(results[results.bond_index == 17].bde_pred.iloc[0])


def test_duplicates():
    results = model.predict(["c1ccccc1"], drop_duplicates=True)
    assert len(results) == 1

    results = model.predict(["c1ccccc1"], drop_duplicates=False)
    assert len(results) == 6


def test_non_canonical_smiles():
    smiles = "CC(=O)OCC1=C\CC/C(C)=C/CC[C@@]2(C)CC[C@@](C(C)C)(/C=C/1)O2"
    assert len(model.predict([smiles])) == 24
