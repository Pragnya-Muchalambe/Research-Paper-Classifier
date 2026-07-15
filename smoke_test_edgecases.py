"""Aggressive edge-case smoke tests for the classifier.

Tests extremely long inputs, unicode, HTML, SQL-like strings, binary-decoded
strings, repeated tokens, and numeric edge values for authors/title/citations.
"""
from research_paper_acceptance_classifier import (
    extract_features,
    generate_dataset,
    predict_with_models,
)
import app as app_module
import random


def run():
    print('\n=== Edge-case smoke tests ===')
    # get cached training bundle (should be fast if already cached)
    df, bundle = app_module.get_training_bundle(sample_size=500, seed=99)

    cases = {}
    # Extremely long text (~50k chars)
    long_words = ['method', 'experiment', 'result', 'analysis']
    cases['huge_text'] = ' '.join(random.choice(long_words) for _ in range(10000))

    # Unicode and emojis
    cases['unicode'] = '我们提出了一种新的方法😊。结果表明优于基线。'

    # HTML-like input
    cases['html'] = '<p>This <strong>paper</strong> shows <em>improvements</em> in <a href="#">results</a>.</p>'

    # SQL-like / code injection text
    cases['sql'] = "'; DROP TABLE papers; -- SELECT * FROM experiments;"

    # Binary-like content decoded
    try:
        raw = bytes(range(256)) * 20
        cases['binary'] = raw.decode('latin1')
    except Exception:
        cases['binary'] = ''.join(chr(i % 256) for i in range(5000))

    # Repeated single token
    cases['repeated'] = 'novel ' * 5000

    # Very high numeric values
    numeric_tests = [
        {'author_count': 0, 'title_length': 0, 'citation_density': 0.0},
        {'author_count': -5, 'title_length': -1, 'citation_density': -0.1},
        {'author_count': 10000, 'title_length': 2000, 'citation_density': 10.0},
    ]

    for name, text in cases.items():
        print(f'\n-- Case: {name} --')
        feats = extract_features(text)
        print('feature sample:', {k: feats[k] for k in ('word_count','readability_score')})
        preds, _ = predict_with_models(text, author_count=3, title_length=10, citation_density=0.05, models=bundle.models)
        print(preds.to_string(index=False))

    # numeric edge cases
    for nt in numeric_tests:
        print('\n-- Numeric test --')
        try:
            preds, _ = predict_with_models('This is a baseline methods paper.', author_count=nt['author_count'], title_length=nt['title_length'], citation_density=nt['citation_density'], models=bundle.models)
            print('nums=', nt, '\n', preds.to_string(index=False))
        except Exception as e:
            print('Numeric test raised exception:', e)

    # Test with a model missing predict_proba: create a dumb wrapper
    class DummyNoProba:
        def predict(self, X):
            return ['Accepted'] * len(X)

    fake_models = dict(bundle.models)
    fake_models['NoProba'] = DummyNoProba()
    print('\n-- Model missing predict_proba --')
    try:
        preds, _ = predict_with_models('We show significant improvements.', 2, 8, 0.1, models=fake_models)
        print(preds.to_string(index=False))
    except Exception as e:
        print('Error when predicting with fake model:', e)

    print('\nEdge-case tests complete')


if __name__ == '__main__':
    run()
