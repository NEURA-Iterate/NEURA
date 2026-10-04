import json
from pathlib import Path

import numpy as np
from scipy import stats

from neura_uq import (
    BayesianStudentTClassifier,
    Measurement,
    kpi_separation,
    leave_one_out,
    rank_kpi_subsets,
)
from neura_uq.cli import main


def test_zero_noise_diag_matches_closed_form_univariate_nig():
    X = np.array([[-1.2], [-0.7], [-0.4], [0.1], [1.0], [1.4], [1.7], [2.3]])
    labels = np.array(["A"] * 4 + ["B"] * 4)
    prior_strength = 2.6
    prior_kappa = 0.01
    classifier = BayesianStudentTClassifier(
        kpis=["x"],
        covariance="diag",
        prior_strength=prior_strength,
        prior_kappa=prior_kappa,
    ).fit(X, labels)
    x = 0.35
    prediction = classifier.predict(Measurement.from_sd([x]))

    grand_mean = X.mean()
    pooled_var = sum(
        np.sum((X[labels == batch] - X[labels == batch].mean()) ** 2)
        for batch in classifier.classes_
    ) / (len(X) - len(classifier.classes_))
    psi0 = (2 + prior_strength) * pooled_var
    expected = {}
    for batch in classifier.classes_:
        xb = X[labels == batch, 0]
        count = len(xb)
        xbar = xb.mean()
        within_ss = np.sum((xb - xbar) ** 2)
        kn = prior_kappa + count
        df = count + prior_strength + 2
        mun = (prior_kappa * grand_mean + count * xbar) / kn
        psin = psi0 + within_ss + prior_kappa * count / kn * (xbar - grand_mean) ** 2
        scale = np.sqrt(psin * (kn + 1) / (kn * df))
        expected[batch] = stats.t.logpdf(x, df, loc=mun, scale=scale)

    for batch in classifier.classes_:
        np.testing.assert_allclose(
            prediction.log_likelihood[batch], expected[batch], atol=1e-10
        )
    posterior = np.exp(list(expected.values()) - np.logaddexp(*expected.values()))
    np.testing.assert_allclose(
        [prediction.probabilities[batch] for batch in classifier.classes_],
        posterior,
        atol=1e-10,
    )


def test_zero_noise_full_matches_scipy_multivariate_t():
    X = np.array(
        [
            [0.1, 1.0],
            [0.3, 1.3],
            [0.2, 0.8],
            [2.0, 3.2],
            [2.2, 2.8],
            [1.8, 3.0],
        ]
    )
    labels = ["A", "A", "A", "B", "B", "B"]
    classifier = BayesianStudentTClassifier(
        kpis=["x", "y"], covariance="full", prior_strength=3.0
    ).fit(X, labels)
    x = np.array([1.1, 2.0])
    prediction = classifier.predict(Measurement.from_sd(x))
    for batch in classifier.classes_:
        expected = stats.multivariate_t.logpdf(
            x,
            loc=classifier.loc_[batch],
            shape=classifier.shape_[batch],
            df=classifier.df_[batch],
        )
        np.testing.assert_allclose(
            prediction.log_likelihood[batch], expected, rtol=1e-10
        )


def test_mc_noise_marginalisation_tiny_and_gaussian_limit():
    X = np.array([[-1.2], [-0.8], [-1.0], [-0.6], [1.0], [1.2], [0.8], [0.6]])
    labels = ["A"] * 4 + ["B"] * 4
    base = BayesianStudentTClassifier(
        kpis=["x"], covariance="diag", n_draws=200_000
    ).fit(X, labels)
    x = 0.25
    noise_free = base.predict(Measurement.from_sd([x]))
    tiny_noise = base.predict(Measurement.from_sd([x], [1e-8]))
    for batch in base.classes_:
        np.testing.assert_allclose(
            tiny_noise.log_likelihood[batch],
            noise_free.log_likelihood[batch],
            atol=1e-6,
        )

    gaussian = BayesianStudentTClassifier(
        kpis=["x"],
        covariance="diag",
        prior_strength=1e6,
        n_draws=200_000,
        random_state=3,
    ).fit(X, labels)
    sigma = 0.5
    prediction = gaussian.predict(Measurement.from_sd([x], [sigma]))
    batch = "A"
    expected = stats.norm.logpdf(
        x,
        loc=gaussian.loc_[batch][0],
        scale=np.sqrt(gaussian.shape_[batch][0, 0] + sigma**2),
    )
    np.testing.assert_allclose(prediction.log_likelihood[batch], expected, atol=0.01)


def test_inflating_measurement_noise_shrinks_that_kpis_influence():
    offsets = np.array([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
    batch_a = np.column_stack([offsets, offsets])
    batch_b = np.column_stack([offsets + 6.0, offsets + 10.0])
    X = np.vstack([batch_a, batch_b])
    labels = ["A"] * len(batch_a) + ["B"] * len(batch_b)
    classifier = BayesianStudentTClassifier(
        kpis=["A_signal", "B_signal"],
        covariance="diag",
        n_draws=50_000,
    ).fit(X, labels)
    sample = [0.0, 6.0]
    precise = classifier.predict(Measurement.from_sd(sample, [0.0, 0.0]))
    noisy_a = classifier.predict(Measurement.from_sd(sample, [30.0, 0.0]))
    assert precise.probabilities["A"] > 0.5
    assert noisy_a.probabilities["A"] < precise.probabilities["A"]


def test_outlier_flag_and_batch_mean_typicality():
    X = np.array(
        [
            [-0.4],
            [-0.2],
            [-0.1],
            [0.1],
            [0.2],
            [0.4],
            [4.6],
            [4.8],
            [4.9],
            [5.1],
            [5.2],
            [5.4],
        ]
    )
    labels = np.array(["A"] * 6 + ["B"] * 6)
    classifier = BayesianStudentTClassifier(kpis=["x"], covariance="diag").fit(
        X, labels
    )
    in_batch_mean = X[labels == "A"].mean(axis=0)
    typical = classifier.predict(Measurement.from_sd(in_batch_mean))
    outlier = classifier.predict(Measurement.from_sd([100.0]))
    assert not typical.outlier
    assert typical.typicality_pvalue["A"] > 0.1
    assert outlier.outlier


def test_reference_noise_is_subtracted_from_new_measurement_noise():
    X = np.array([[0.0], [1.0], [2.0], [4.0], [5.0], [6.0]])
    labels = ["A", "A", "A", "B", "B", "B"]
    measurements = [Measurement.from_sd([value], [0.2]) for value in X[:, 0]]
    classifier = BayesianStudentTClassifier(kpis=["x"], covariance="diag").fit(
        X, labels, reference_measurements=measurements
    )
    _, excess = classifier.excess_noise(Measurement.from_sd([3.0], [0.2]))
    np.testing.assert_allclose(excess, np.zeros((1, 1)), atol=1e-12)


def test_from_runs_matches_log_delta_method_for_small_variance():
    runs = np.linspace(1.98, 2.02, 20)[:, None]
    from_runs = Measurement.from_runs(runs)
    from_cov = Measurement.from_cov(runs.mean(axis=0), np.cov(runs, rowvar=False))
    _, run_cov = from_runs.transformed(["log"])
    _, delta_cov = from_cov.transformed(["log"])
    np.testing.assert_allclose(run_cov, delta_cov, rtol=0.05)


def test_leave_one_out_and_subset_ranking_find_informative_kpi():
    rng = np.random.default_rng(23)
    labels = np.repeat(["B1", "B2", "B3"], 7)
    signal = np.concatenate([rng.normal(center, 0.2, 7) for center in (-4.0, 0.0, 4.0)])
    noise = rng.normal(0.0, 1.0, len(labels))
    X = np.column_stack([signal, noise])
    loo = leave_one_out(
        X,
        labels,
        kpis=["signal", "noise"],
        covariance="diag",
    )
    assert loo["accuracy"] == 1.0
    assert loo["log_loss"] < loo["baseline_log_loss"]

    ranked = rank_kpi_subsets(
        X,
        labels,
        ["signal", "noise"],
        max_size=2,
        covariance="diag",
    )
    assert [result["log_loss"] for result in ranked] == sorted(
        result["log_loss"] for result in ranked
    )
    assert "signal" in ranked[0]["kpis"]
    separation = kpi_separation(X, labels, ["signal", "noise"])
    assert separation["signal"] > separation["noise"]


def test_cli_smoke_for_sd_and_perturbed_run_samples_with_loo(tmp_path, capsys):
    examples = Path(__file__).resolve().parents[1] / "examples"
    reference = examples / "reference_example.csv"
    sample = examples / "sample_example.csv"
    assert (
        main(
            [
                "--reference",
                str(reference),
                "--sample",
                str(sample),
                "--transforms",
                "logit,logit,log,log",
                "--covariance",
                "diag",
                "--loo",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Predicted batch:" in output
    assert "Leave-one-out metrics:" in output
    assert "KPI separation:" in output

    runs = tmp_path / "runs.csv"
    runs.write_text(
        "si_frac,porosity,d50,agglom\n"
        "0.118,0.27,5.5,0.55\n"
        "0.12,0.28,5.6,0.57\n"
        "0.124,0.275,5.58,0.56\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--reference",
                str(reference),
                "--sample",
                str(runs),
                "--transforms",
                "logit,logit,log,log",
                "--covariance",
                "diag",
                "--loo",
                "--json",
            ]
        )
        == 0
    )
    output_json = json.loads(capsys.readouterr().out)
    assert set(output_json) == {
        "probabilities",
        "pvalues",
        "flags",
        "kpi_evidence",
        "loo",
        "kpi_separation",
    }
    assert output_json["loo"]["accuracy"] >= 0.0


def test_typicality_pvalues_are_calibrated_for_small_batches():
    rng = np.random.default_rng(0)
    p, class_count, samples_per_class, trials = 4, 3, 7, 300

    for covariance in ("full", "diag"):
        pvalues = []
        for _ in range(trials):
            if covariance == "full":
                A = rng.normal(size=(p, p))
                S = A @ A.T / p + 0.5 * np.eye(p)
            else:
                S = np.diag(rng.uniform(0.5, 1.5, p))
            means = rng.normal(0, 1, (class_count, p))
            X = np.vstack(
                [
                    rng.multivariate_normal(means[k], S, size=samples_per_class)
                    for k in range(class_count)
                ]
            )
            labels = np.repeat(np.arange(class_count), samples_per_class)
            classifier = BayesianStudentTClassifier(
                kpis=[f"kpi_{j}" for j in range(p)],
                covariance=covariance,
                prior_strength=2.0,
                n_draws=1,
            ).fit(X, labels)
            true_batch = rng.integers(class_count)
            x = rng.multivariate_normal(means[true_batch], S)
            prediction = classifier.predict(Measurement.from_sd(x))
            pvalues.append(prediction.typicality_pvalue[true_batch])

        rejection_rate = np.mean(np.asarray(pvalues) < 0.05)
        print(f"{covariance} typicality rejection rate: {rejection_rate:.4f}")
        assert 0.02 <= rejection_rate <= 0.10
