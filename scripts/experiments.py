"""Define the 36 modality/scope experiments and cross-category combinations."""
from itertools import combinations
from config import (
    TARGETS, CATEGORY_FILES, RELEVANT_CATEGORIES, ALL_BEHAVIOR_FILES,
    BEHAVIOR_SCOPES, NON_SYMPTOM_FILES
)


def unique_files(files):
    return sorted(set(files))


def files_for_categories(categories):
    files = []
    for category in categories:
        files.extend(CATEGORY_FILES[category])
    return unique_files(files)


def files_for_scope(target_key, scope):
    target_file = TARGETS[target_key]['file']
    target_category = TARGETS[target_key]['category']

    if scope == 'all':
        files = ALL_BEHAVIOR_FILES
    elif scope == 'relevant':
        files = files_for_categories(RELEVANT_CATEGORIES[target_key])
    elif scope == 'category':
        files = files_for_categories([target_category])
    elif scope == 'non-symptom':
        files = NON_SYMPTOM_FILES
    else:
        raise ValueError(f'Unknown behavior scope: {scope}')

    # The target questionnaire is never allowed to predict its own total score.
    return [f for f in unique_files(files) if f != target_file]


def build_core_experiments():
    """Four imaging models + 16 behavioral + 16 multimodal = 36 models."""
    experiments = []

    for target_key, target in TARGETS.items():
        experiments.append({
            'experiment_name': f'{target_key}__imaging_only',
            'target_key': target_key,
            'target_label': target['label'],
            'target_file': target['file'],
            'target_column': target['column'],
            'target_category': target['category'],
            'valid_min': target['valid_min'],
            'valid_max': target['valid_max'],
            'input_type': 'imaging_only',
            'behavior_scope': 'none',
            'behavior_files': [],
            'include_behavior': False,
            'include_imaging': True,
        })

        for input_type in ['behavioral_only', 'multimodal']:
            for scope in BEHAVIOR_SCOPES:
                experiments.append({
                    'experiment_name': f'{target_key}__{input_type}__{scope}',
                    'target_key': target_key,
                    'target_label': target['label'],
                    'target_file': target['file'],
                    'target_column': target['column'],
                    'target_category': target['category'],
                    'valid_min': target['valid_min'],
                    'valid_max': target['valid_max'],
                    'input_type': input_type,
                    'behavior_scope': scope,
                    'behavior_files': files_for_scope(target_key, scope),
                    'include_behavior': True,
                    'include_imaging': input_type == 'multimodal',
                })
    return experiments


def build_cross_category_experiments():
    """
    For each target, start with its own category and add every subset of the
    other three categories. Four targets x eight category combinations = 32.
    """
    experiments = []
    all_categories = list(CATEGORY_FILES)

    for target_key, target in TARGETS.items():
        base_category = target['category']
        outside = [c for c in all_categories if c != base_category]

        for number_added in range(len(outside) + 1):
            for added_tuple in combinations(outside, number_added):
                included = [base_category, *added_tuple]
                label = ' + '.join(c.title() for c in included)
                suffix = '__'.join(included)
                files = files_for_categories(included)
                files = [f for f in files if f != target['file']]

                experiments.append({
                    'experiment_name': f'{target_key}__cross__{suffix}',
                    'target_key': target_key,
                    'target_label': target['label'],
                    'target_file': target['file'],
                    'target_column': target['column'],
                    'target_category': base_category,
                    'valid_min': target['valid_min'],
                    'valid_max': target['valid_max'],
                    'base_category': base_category,
                    'added_categories': list(added_tuple),
                    'included_categories': included,
                    'combination_label': label,
                    'behavior_files': files,
                })
    return experiments


CORE_EXPERIMENTS = build_core_experiments()
CROSS_CATEGORY_EXPERIMENTS = build_cross_category_experiments()


if __name__ == '__main__':
    print(f'Core experiments: {len(CORE_EXPERIMENTS)}')
    print(f'Cross-category experiments: {len(CROSS_CATEGORY_EXPERIMENTS)}')
    for exp in CORE_EXPERIMENTS:
        print(exp['experiment_name'])
