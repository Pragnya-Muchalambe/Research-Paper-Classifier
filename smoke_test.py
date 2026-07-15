"""Non-interactive smoke tests for Research Paper Acceptance Classifier.

Covers: basic health, input validation, functional correctness,
model behavior checks, performance-ish checks, output validation,
and feature extraction sanity.
"""
import time
from research_paper_acceptance_classifier import (
    extract_features,
    generate_dataset,
    predict_with_models,
    train_models,
)

import app as app_module


def print_header(title: str):
    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)


def basic_app_health():
    print_header('1) Basic ML Module Health')
    df = generate_dataset(n_samples=300, seed=123)
    print('Generated dataset rows:', len(df))
    bundle = train_models(df, seed=123)
    print('Trained models:', list(bundle.models.keys()))


def input_validation():
    print_header('2) Input Validation')
    cases = {
        'empty': '',
        'short': 'This paper studies experiments.',
        'long': ' '.join(['This is a long sentence about methods and experiments.'] * 50),
        'weird': '@@@ 123 ??? ### $$$',
    }
    for name, text in cases.items():
        feats = extract_features(text)
        print(f'Case={name} -> features sample:', {k: feats[k] for k in ['word_count', 'avg_sentence_length', 'readability_score']})


def functional_correctness():
    print_header('3) Functional Correctness')
    strong = 'We propose a novel transformer-based framework that significantly outperforms state-of-the-art methods on benchmarks. Extensive experiments demonstrate superior results.'
    weak = 'This paper studies some basic ideas and compares a few existing methods with marginal differences and limited experiments.'
    neutral = 'We explore different techniques and evaluate them on several datasets to understand relative behavior.'

    df = generate_dataset(n_samples=600, seed=7)
    bundle = train_models(df, seed=7)

    for label, text in [('Strong', strong), ('Weak', weak), ('Neutral', neutral)]:
        preds, feats = predict_with_models(text, author_count=3, title_length=10, citation_density=0.05, models=bundle.models)
        print(f'\n{label} paper predictions:')
        print(preds.to_string(index=False))


def model_behavior_checks():
    print_header('4) Model Behavior Checks')
    # change sample size
    df1 = generate_dataset(n_samples=300, seed=10)
    b1 = train_models(df1, seed=10)
    preds1, _ = predict_with_models('We propose a novel method.', 3, 10, 0.05, b1.models)

    df2 = generate_dataset(n_samples=1200, seed=10)
    b2 = train_models(df2, seed=10)
    preds2, _ = predict_with_models('We propose a novel method.', 3, 10, 0.05, b2.models)

    print('Preds (small sample):')
    print(preds1.to_string(index=False))
    print('Preds (large sample):')
    print(preds2.to_string(index=False))

    # change seed
    df3 = generate_dataset(n_samples=600, seed=20)
    b3 = train_models(df3, seed=20)
    preds3, _ = predict_with_models('We propose a novel method.', 3, 10, 0.05, b3.models)
    print('Preds (seed 20):')
    print(preds3.to_string(index=False))


def performance_check_and_cache():
    print_header('5) Performance & Cache Checks')
    # call app cache function twice and measure time
    t0 = time.time()
    df_a, bundle_a = app_module.get_training_bundle(sample_size=700, seed=42)
    first = time.time() - t0
    t1 = time.time()
    df_b, bundle_b = app_module.get_training_bundle(sample_size=700, seed=42)
    second = time.time() - t1
    print(f'First call (should train): {first:.2f}s, second call (should be cached): {second:.2f}s')

    # change parameter -> should retrain
    t2 = time.time()
    df_c, bundle_c = app_module.get_training_bundle(sample_size=800, seed=42)
    changed = time.time() - t2
    print(f'Call with changed sample_size (retrain expected): {changed:.2f}s')


def output_validation():
    print_header('6) Output Validation')
    df = generate_dataset(n_samples=400, seed=5)
    bundle = train_models(df, seed=5)
    preds, _ = predict_with_models('We propose a novel method.', 3, 10, 0.05, bundle.models)
    print(preds.to_string(index=False))
    # check confidence values
    for c in preds['Confidence (%)']:
        assert c >= 0.0 and c <= 100.0, f'Bad confidence value: {c}'
    print('Confidence values OK')


def feature_extraction_sanity():
    print_header('7) Feature Extraction Sanity')
    text = 'Novel framework with strong results and extensive experiments demonstrating improvements.'
    feats = extract_features(text)
    print('Features:', feats)


def run_all():
    basic_app_health()
    input_validation()
    functional_correctness()
    model_behavior_checks()
    performance_check_and_cache()
    output_validation()
    feature_extraction_sanity()


if __name__ == '__main__':
    run_all()
