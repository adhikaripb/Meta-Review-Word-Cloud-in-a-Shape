# Generates wordcloud for a list of given terms from the given multi-reference .ris file.

import os, re, rispy, csv, string, time, random, json, ast
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict, Counter
from PIL import Image
import numpy as np
import pandas as pd

import argparse
# …existing imports remain…

def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line options so the script can run non-interactively."""
    parser = argparse.ArgumentParser(
        description="Generate word-clouds from literature reference files."
    )
    parser.add_argument("--max-font-size", type=int, default=700,
                        help="Maximum font size for the largest word.")
    parser.add_argument("--shape-mask-path", type=str, default="",
                        help="PNG mask that defines the cloud’s shape.")
    parser.add_argument("--exclude-words", type=str, default="",
                        help="Comma-separated list of words to exclude.")
    parser.add_argument("--inclusion-terms-file", type=str, default="",
                        help="Path to a text file containing inclusion terms.")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="Treat search terms as case-sensitive.")
    parser.add_argument("--medium-freq-threshold", type=float, default=None,
                        help="Medium frequency threshold for colour mapping.")
    parser.add_argument("--minimum-freq-threshold", type=float, default=None,
                        help="Minimum frequency threshold for colour mapping.")
    parser.add_argument("--show-plot", action="store_true",
                        help="Display the resulting figure as well as saving it.")
    parser.add_argument("--constraints", type=str, default="",
                        help="Comma-separated constraint terms.")
    parser.add_argument("--constraints-listoflists", action="store_true",
                        help="Interpret --constraints as a ‘list-of-lists’.")
    parser.add_argument("--meta-constraints", type=str, default="",
                        help="Stringified list-of-lists for meta-constraints.")
    parser.add_argument("--mutual-inclusivity", action="store_true",
                        help="Require all constraint terms to co-occur.")
    return parser.parse_args()

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # If running interactively
    script_dir = os.getcwd()
base_dir = script_dir
print(f"Please note that output will be saved at {base_dir}/outputs")

def random_color_func():
    # Randomly pick a hue (from 0 to 360)
    hue = random.randint(0, 360)
    # Set a high saturation to avoid gray colors. Usually, above 60% is safe.
    saturation = random.randint(60, 100)
    # Lightness can be set to any value that does not result in white, black, or gray.
    lightness = 47  # You can adjust this value based on your preference

    # Return an HSL color excluding gray
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def parse_user_input_list_of_lists(user_input):
    try:
        # Replace single quotes with double quotes for JSON compatibility
        formatted_input = user_input.replace("'", '"')

        # Load the input as a JSON-like list
        parsed_list = json.loads(formatted_input)

        # Validate that the parsed object is a list of lists
        if not isinstance(parsed_list, list) or not all(isinstance(sublist, list) for sublist in parsed_list):
            raise ValueError("Input should be a list of lists.")

        # Further validate that all elements in each sublist are strings
        for sublist in parsed_list:
            if not all(isinstance(item, str) for item in sublist):
                raise ValueError("All items in the sublists should be strings.")

        return parsed_list
    except json.JSONDecodeError as e:
        print(f"An error occurred: {str(e)}")
        return 'Not a list'
    except ValueError as e:
        print(str(e))
        return 'Not a list'
    except AttributeError as e:
        print(f"\nThe given list of list is already valid.")
        return user_input

def parse_user_input_simple_list(user_input):
    try:
        # First, try to parse it as JSON
        parsed_list = json.loads(user_input)
    except json.JSONDecodeError:
        # If JSON parsing fails, try parsing it as a Python literal
        try:
            parsed_list = ast.literal_eval(user_input)
            if not isinstance(parsed_list, list):
                raise ValueError("Input should be a list formatted as a JSON or a Python list literal.")
        except (SyntaxError, ValueError) as e:
            print(f"An error occurred while parsing as Python literal: {str(e)}")
            return []

        # Check if it's a list and all items are strings
    if not isinstance(parsed_list, list):
        print("Input should be a list.")
        return []
    if not all(isinstance(item, str) for item in parsed_list):
        print("All items in the list should be strings.")
        return []

    return parsed_list

def custom_color_func(word, font_size, position, orientation, random_state=None, font_path=None, medium_freq_word_threshold= 0.8, minimum_freq_word_threshold = 0.4, max_font_size=700):
    print(f"Word: '{word}', Font Size: {font_size}\n font-to-max_font ratio: {font_size/max_font_size}")
    if font_size > max_font_size * medium_freq_word_threshold:  # Adjust the threshold based on max_font_size
        saturation = int(20 + 80 * (font_size / max_font_size))  # Adjust saturation from 20% to 100%
        # return f"hsl(123, {saturation}%, 47%)" # a shade of green for high frequency words
        return random_color_func() # pick random color except any shade of gray
    elif font_size > max_font_size * minimum_freq_word_threshold:
        return "gray" # gray for medium frequency words
    else:
        return "lightgray" # light gray for lowest frequency words

def read_ris_file(file_path):
    try:
        with open(file_path, 'r') as file:
                entries = rispy.load(file)
    except Exception as e:
        print(f'Error occurred: {e}. Trying cleaning, saving and loading the ris file')
        clean_ris_file_path = os.path.join(os.path.dirname(file_path), "cleaned_" + os.path.basename(file_path))
        clean_ris_file(file_path, clean_ris_file_path)
        with open(clean_ris_file_path, 'r') as file:
            entries = rispy.load(file)
    return entries

def save_word_frequencies_to_csv(word_freq, csv_file_path):
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Word', 'Frequency'])
            for word, freq in word_freq.items():
                writer.writerow([word, freq])
    except Exception as e:
        print(f'error encountered: {e}\nShortening file name and moving forward.')
        csv_file_name = os.path.basename(csv_file_path).split(".")[0]
        csv_file_newname = f'{csv_file_name[:80]}...{csv_file_name[-90:]}.csv'
        print(f'shortened file name: {csv_file_newname}')
        csv_file_newpath = f'{os.path.dirname(csv_file_path)}/{csv_file_newname}'
        with open(csv_file_newpath, 'w', newline='', encoding='utf-8') as csvnewfile:
            writer = csv.writer(csvnewfile)
            writer.writerow(['Word', 'Frequency'])
            for word, freq in word_freq.items():
                writer.writerow([word, freq])

def save_word_frequencies_to_csv_w_headers(word_data, csv_file_path, all_headers):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        headers = ['Word'] + all_headers + ['Total Count', 'Associated Constraints']
        writer.writerow(headers)
        for word, data in word_data.items():
            associated_constraints = '; '.join(
                [' | '.join(sorted(constraint_list)) for constraint_list in data['constraints'].values()])
            row = [word] + [data['values'].get(header, 0) for header in all_headers] + [data['total_count'],
                                                                                        associated_constraints]
            writer.writerow(row)

def save_word_frequencies_to_csv_w_constraints(word_freq_dict, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Frequency', 'Files', 'Constraints'])
        for word, data in word_freq_dict.items():
            constraints = [' | '.join(constraint) for constraint in data['constraints']]  # Join each list into a string
            writer.writerow(
                [word, data['count'], ', '.join(data['files']), '; '.join(constraints)])  # Join lists with semicolon

def extract_constraints(filename, choice = 'c'):
    # This regex looks for "constrain-" followed by any characters and captures text within brackets
    match = re.search(r'constrain-\[(.*?)\]', filename)
    if choice == 'c':
        if match:
            return match.group(1).split(', ')
        return []
    else:
        if match:
            return [term.strip() for term in match.group(1).split(',')]  # Return list of constraints
        return [filename]  # Use filename if no constraints found

def standardize_phrase(word, wordcase='insensitive'):
    if wordcase == 'insensitive':
        return re.sub(r'[^\w\s]+', '', word).lower()  # Allow spaces within phrases
    else:
        return re.sub(r'[^\w\s]+', '', word)

def standardize_word(word, wordcase='insensitive'):
    def handle_complex_case(base, part):
        # Extract numeric parts
        numeric_parts = re.findall(r'\d+', part)
        return [base + num_part for num_part in numeric_parts] if numeric_parts else [base]

    # Normalize case if necessary
    if wordcase == 'insensitive':
        word = word.lower()

    # Replace hyphens with spaces, then clean up the word and split based on '/' and '('
    cleaned_word = re.sub(r'-', ' ', word)
    cleaned_word = re.sub(r'[^\w\s/]+', '', cleaned_word).strip()
    split_parts = re.split(r'[/(]', cleaned_word)
    parts = []
    for part in split_parts:
        parts = parts + re.split(r'[/]', part)

    expanded_words = []
    if parts:
        # Find the base from the first part
        first_part = parts[0]
        base_match = re.match(r'^(.*?)(\d+)?$', first_part)
        base = base_match.group(1) if base_match else first_part
        for part in parts:
            expanded_words.extend(handle_complex_case(base, part))

    return expanded_words

def replace_hyphen(text):
    # This pattern ensures hyphens in 'Agamous-like' and 'pak-choi' or their case variations are not replaced
    # Uses case-insensitive flag directly in the pattern with `(?i)` instead of in re.sub function
    pattern = r'(?i)-(?!(?<=\bagamous-)\blike\b)(?!(?<=\bpak-)\bchoi\b)(?!(?<=\bAS-)\bMADS\b)'
    return re.sub(pattern, ' ', text)

def contains_constraints(content, constraints, exclude_words):
    content_parts = content.split()  # Split the content into words for processing [seems like this line is unnecessary here]

    # Convert constraints to a set of tuples if they are lists, or a set of the original items otherwise
    if isinstance(constraints, set):
        constraint_set = constraints
    else:
        constraint_set = set(
            tuple(constraint) if isinstance(constraint, list) else constraint for constraint in constraints)

    # Build patterns from exclude_words to identify exact phrases to ignore
    exclude_patterns = exclude_words if isinstance(exclude_words, set) else set(exclude_words)
    # constraint_set = constraints if isinstance(constraints, set) else set(constraints)

    # Create a modified version of content that excludes the `exclude_words` completely
    modified_content = content.lower()
    # print(f'exclude_pattern: {exclude_words}')
    for pattern in exclude_patterns:
        # Use regex to replace exact phrases, considering word boundaries
        modified_content = re.sub(r'\b' + re.escape(pattern.lower()) + r'\b', "", modified_content)
    # if 'agl62' in modified_content: # for debugging
    #     print(f'modified_content: {modified_content}')

    # Check if modified content still contains any of the constraints
    for constraint in constraint_set:
        # Use regex to check for exact word matches
        # Here constraint needs to be converted back to string if it's a tuple
        constraint_pattern = ' '.join(constraint) if isinstance(constraint, tuple) else constraint
        if re.search(r'\b' + re.escape(constraint_pattern) + r'\b', modified_content):
            return True  # Constraint appears outside of excluded phrases as a whole word

    return False  # No valid occurrence of constraints found

def contains_whole_word(content, word):
    return bool(re.search(r'\b' + re.escape(word) + r'\b', content))

def create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, exclude_words_phrase = 'n', constraints_listoflists = 'n', constraints=None, medium_freq_word_threshold = None, minimum_freq_word_threshold = None, exclude_words=None,
                      include_only=None, mutual_inclusivity = None, wordcase='insensitive', patterns=None, show_plot = 'y', min_occurrence=1, max_font_size = 700):
    csv_file_path = os.path.join(out_dir, csv_file)
    word_cloud_file_path = os.path.join(out_dir, word_cloud_file)

    # print(f'exclude_words_pre: {exclude_words}')
    if exclude_words is None:
        exclude_words = set()
    else:
        # exclude_words = set(standardize_word(word, wordcase=wordcase) for word in exclude_words)
        exclude_words_set = set()
        for word in exclude_words:
            for standardized_word in standardize_word(word, wordcase=wordcase):
                exclude_words_set.add(standardized_word)
        print(f'exclude_word_set: {exclude_words_set}')

    if include_only is not None:
        # include_only = set(standardize_phrase(word, wordcase=wordcase) for word in include_only)
        # print(f'include_only = {include_only}')
        include_only_groups = [[standardize_phrase(term.strip(), wordcase=wordcase) for term in group] for group in include_only]
        print(f'include_only = {include_only}')
        print(f'include_only groups = {include_only_groups}')
    word_freq = Counter()
    first_occurrence = {} # To track and merge count cases like 'StMADS11' and 'STMADS11'

    for ref in references:
        title = ref.get('title', '')
        abstract = ref.get('abstract', '')
        content = f"{title} {abstract}".lower() if wordcase == 'insensitive' else f"{title} {abstract}"

        # print(f'nascent content: {content}')
        # Split the content into words and also get word pairs for two-word phrases
        # words = content.split()
        cleaned_content = replace_hyphen(content)
        # cleaned_content = [re.sub(r'[():,/]', ' ', word).lower() if wordcase == 'insensitive' else word for word in cleaned_content]
        cleaned_content = re.sub(r"[:(),/]", ' ', replace_hyphen(content))

        # print(f'cleaned content: {cleaned_content}')
        # Split the word into individual words
        words = cleaned_content.split()
        word_pairs = [' '.join(pair) for pair in zip(content.split(), content.split()[1:])]

        # # check if the reference content contains phrases and continue only if it doesn't
        # if exclude_words_phrase == 'y':
        #     for phrase in word_pairs:
        #         standardized_phrase = standardize_phrase(phrase, wordcase=wordcase)
        #         if standardized_phrase not in exclude_words:
        #             print(f'Title without exclusion_phrases: {title}')
        #             continue

        # For constraints, consider both single words and two-word phrases
        if constraints:
            if constraints_listoflists == 'y':
                for sublist in constraints:
                    constraints_phrases = set()
                    for constraint in sublist:
                        if ' ' in constraint:  # Check if constraint is a phrase
                            constraints_phrases.add(constraint.lower() if wordcase == 'insensitive' else constraint)
            elif constraints_listoflists == 'n':
                constraints_phrases = set()
                for constraint in constraints:
                    if ' ' in constraint:  # Check if constraint is a phrase
                        constraints_phrases.add(constraint.lower() if wordcase == 'insensitive' else constraint)
        # print(f'\ncontent: {content}')
        # print(f'constraints: {constraints}')
        # print(f'exclude_words: {exclude_words}')
        # print(f'contains_constraints(cleaned_content, constraints_phrases, exclude_words): {contains_constraints(cleaned_content, constraints, exclude_words)}')

        if exclude_words_phrase == 'y' and contains_constraints(cleaned_content, constraints, exclude_words) == False:
            # if 'AGL62' in cleaned_content: # for debugging
            #     print(f'Omitted reference: {title}')
            continue

        # Remove exclude phrases from the content
        full_exclude_words = exclude_words.copy()
        for item in exclude_words:
            full_exclude_words.append(f'{item[0].upper()}{item[1:]}')
            full_exclude_words.append(item.upper())
        # Build patterns from exclude_words to identify exact phrases to ignore
        exclude_patterns = full_exclude_words if isinstance(full_exclude_words, set) else set(full_exclude_words)
        # constraint_set = constraints if isinstance(constraints, set) else set(constraints)
        # Create a modified version of content that excludes the `exclude_words` completely
        modified_content = cleaned_content
        # print(f'exclude_pattern: {exclude_words}')
        for pattern in exclude_patterns:
            # Use regex to replace exact phrases, considering word boundaries
            modified_content = re.sub(r'\b' + re.escape(pattern) + r'\b', "", modified_content)


        # else:
        #     content_wo_exclude_phr
        # print(f'Reference under consideration now: {title}')
        # Compile patterns for efficiency
        if patterns:
            compiled_patterns = [
                re.compile(pattern, re.IGNORECASE) if wordcase == 'insensitive' else re.compile(pattern) for pattern in
                patterns] if patterns else []
            # print(f'compiled_patterns: {compiled_patterns}')
            counted_words = set()

            for word in words:
                for split_word in word.split():
                    # print(f'word after split: {split_word}')
                    standardized_word = standardize_word(split_word, wordcase=wordcase)[0]  # Assuming standardize_word returns a list
                    # print(f'standardized split word: {standardized_word}')
                    # print(f'standardized_word in counted_words: {standardized_word in counted_words}')
                    # print(f'standardized_word in exclude_words_set: {standardized_word in exclude_words_set}')
                    # Skip words if it is already counted in the ongoing reference content
                    if standardized_word in counted_words or standardized_word in exclude_words_set:
                        # print(f'after first screening: {standardized_word}')
                        continue
                    # Apply pattern matching
                    # print(f'patterns being checked: {patterns}')

                    if constraints_listoflists == 'y':
                        # Apply mutual inclusivity/exclusivity constraints
                        if mutual_inclusivity in [None, 'n']:
                            if constraints and not any(
                                    any(contains_whole_word(modified_content, constraint) for constraint in sublist) for sublist
                                    in constraints) and not any(
                                    any(contains_whole_word(modified_content, constraint) for constraint in constraints_phrases)
                                    for constraints_phrases in constraints):
                                continue
                        if mutual_inclusivity == 'y':
                            if constraints and not all(
                                    any(contains_whole_word(modified_content, constraint) for constraint in sublist) for sublist
                                    in constraints):
                                continue
                        # This line ensures all phrases in `constraints_phrases` must also be present in `content`
                        if constraints_phrases and not all(
                                any(contains_whole_word(modified_content, constraint) for constraint in constraints_phrases) for
                                constraints_phrases in constraints):
                            continue

                    if constraints_listoflists == 'n':
                        if mutual_inclusivity in [None, 'n']:
                            # print(f'fetching mutually exclusive terms for {constraints}')
                            if constraints and not any(
                                    contains_whole_word(modified_content, constraint) for constraint in constraints) and not any(
                                    contains_whole_word(modified_content, constraint) for constraint in constraints_phrases):
                                continue
                        elif mutual_inclusivity == 'y':
                            # print(f'fetching mutually inclusive terms for {constraints}')
                            # Check if all elements of `constraints` are in `content`
                            if constraints and not all(
                                    contains_whole_word(modified_content, constraint) for constraint in constraints):
                                continue
                            # This line ensures all phrases in `constraints_phrases` must also be present in `content`
                            if constraints_phrases and not all(
                                    contains_whole_word(modified_content, constraint) for constraint in constraints_phrases):
                                continue

                    # Skip words based on patterns
                    if patterns is not None and not any(pattern.search(standardized_word) for pattern in compiled_patterns):
                        # print(f'after second screening: {standardized_word}')
                        continue
                    # Skip words in the exclusion list
                    if exclude_words is not None and standardized_word in exclude_words_set:
                        continue
                    # Determine the first occurrence for case preservation in counting
                    if standardized_word.lower() not in first_occurrence and not standardized_word.islower():
                        first_occurrence[standardized_word.lower()] = standardized_word

                    # Count using the original casing of the first occurrence, skip if entirely lowercase
                    if not standardized_word.islower():
                        word_freq[first_occurrence.get(standardized_word.lower(), standardized_word)] += 1
                        counted_words.add(standardized_word)
                        print(f'first_occurrence: {first_occurrence}')
                        print(f'word = {word}(found pattern: ')
                        any(print(f"{pattern.pattern}' matches with '{standardized_word}'") for pattern in
                            compiled_patterns if pattern.search(standardized_word))
                        print(f'standardized_word = {standardized_word}')
                        print(f'title for above word/phrase = {title}')
                        if standardized_word == 'STMADS11':
                            break
                    else:
                        print(f"Skipping lowercase: {standardized_word}")

        else:
            counted_words = set()
            already_counted_groups = set()  # To ensure we don't count the same group more than once per reference

            for phrase in words + word_pairs:
                standardized_phrase = standardize_phrase(phrase, wordcase=wordcase)
                # print(f'standardized_phrase = {standardized_phrase}')
                if standardized_phrase in counted_words or standardized_phrase in exclude_words:
                    continue
                # if include_only and standardized_phrase not in include_only:
                #     continue
                include_only_matched = False
                # print(f'include_only_groups: {include_only_groups}')
                if include_only_groups:
                    for group in include_only_groups:
                        if standardized_phrase in group:
                            group_base_term = group[0]
                            if group_base_term not in already_counted_groups:
                                if constraints_listoflists == 'y':
                                    # Apply mutual inclusivity/exclusivity constraints
                                    if mutual_inclusivity in [None, 'n']:
                                        if constraints and not any(
                                                any(contains_whole_word(modified_content, constraint) for constraint in sublist)
                                                for sublist in constraints) and not any(any(
                                                contains_whole_word(modified_content, constraint) for constraint in
                                                constraints_phrases) for constraints_phrases in constraints):
                                            continue
                                    if mutual_inclusivity == 'y':
                                        if constraints and not all(
                                                any(contains_whole_word(modified_content, constraint) for constraint in sublist)
                                                for sublist in constraints):
                                            continue
                                    # This line ensures all phrases in `constraints_phrases` must also be present in `content`
                                    if constraints_phrases and not all(any(
                                            contains_whole_word(modified_content, constraint) for constraint in
                                            constraints_phrases) for constraints_phrases in constraints):
                                        continue
                                if constraints_listoflists == 'n':
                                    if mutual_inclusivity in [None, 'n']:
                                        # print(f'fetching mutually exclusive terms for {constraints}')
                                        if constraints and not any(
                                                contains_whole_word(modified_content, constraint) for constraint in
                                                constraints) and not any(
                                                contains_whole_word(modified_content, constraint) for constraint in
                                                constraints_phrases):
                                            continue
                                    elif mutual_inclusivity == 'y':
                                        # print(f'fetching mutually inclusive terms for {constraints}')
                                        # Check if all elements of `constraints` are in `content`
                                        if constraints and not all(
                                                contains_whole_word(modified_content, constraint) for constraint in
                                                constraints):
                                            continue
                                        # This line ensures all phrases in `constraints_phrases` must also be present in `content`
                                        if constraints_phrases and not all(
                                                contains_whole_word(modified_content, constraint) for constraint in
                                                constraints_phrases):
                                            continue

                                word_freq[group_base_term] += 1
                                already_counted_groups.add(group_base_term)
                                include_only_matched = True
                                print(f'filtered term = {phrase}')
                                # print(f'filtered title = {title}')
                                print(f'cleaned_content: {cleaned_content}')
                                print(f'word_freq[{group_base_term}] = {word_freq[group_base_term]}')
                                counted_words.add(standardized_phrase)  # Ensuring this phrase isn't counted again
                            break  # Stop checking other groups once a match is found

                if include_only_matched:
                    continue
                    # print(f'filtered word/phrase = {phrase}')
                    # word_freq[standardized_phrase] += 1
                    # counted_words.add(standardized_phrase)
                    # print(f'filtered word/phrase = {phrase}')
                    # print(f'title for above word/phrase = {title}')

    filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_occurrence}
    save_word_frequencies_to_csv(filtered_word_freq, csv_file_path)

    if shape_mask_path is None:
        # Create a full mask (every pixel is available for words)
        mask = np.zeros((1000, 1000), dtype=np.uint8)  # Size should match 'width' and 'height' of the WordCloud
    else:
        mask_image = Image.open(shape_mask_path).convert('L')
        mask = np.array(mask_image)

    if constraints != None:
        print(f'constraints used for screening: {constraints}')

    print(f'filtered_word_freq= {filtered_word_freq}')
    print(f'\nGenerating wordcloud image for the collected terms. Please hold on...')

    if medium_freq_word_threshold == None or minimum_freq_word_threshold == None:
        wordcloud = WordCloud(
            width=1000,
            height=1000,
            background_color='white',
            mask=mask,
            max_words=500,
            # relative_scaling=0, # Ensure sizes are only based on the provided frequencies
            # min_font_size=100, # Generates erraneous wordcloud if frequencies of all terms are same
            max_font_size= max_font_size, # Generates erraneous wordcloud if frequencies of all terms are same
            scale=10,
        ).generate_from_frequencies(filtered_word_freq)

        # Printing each word's font size after generation
        for (word, freq), font_size, position, orientation, color in wordcloud.layout_:
            print(f"Word: '{word}' with frequency {freq} has font size: {font_size}")

    else:
        wordcloud = WordCloud(
            width=1000,
            height=1000,
            background_color='white',
            mask=mask,
            max_words=500,
            # relative_scaling=0, # Ensure sizes are only based on the provided frequencies
            # min_font_size=100, # Generates erraneous wordcloud if frequencies of all terms are same
            max_font_size=max_font_size,  # Generates erraneous wordcloud if frequencies of all terms are same
            scale=10,
            color_func=lambda word, font_size, position, orientation, random_state, font_path: custom_color_func(
                word, font_size, position, orientation, random_state, font_path,
                medium_freq_word_threshold=medium_freq_word_threshold,
                minimum_freq_word_threshold=minimum_freq_word_threshold,
                max_font_size=max_font_size)
        ).generate_from_frequencies(filtered_word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    try:
        plt.savefig(word_cloud_file_path, dpi=600)
    except Exception as e:
        print(f'Error occurred while saving file: {e}\nSaving with modified file name')
        word_cloud_file_path = f'{os.path.dirname(word_cloud_file_path)}/{os.path.basename(word_cloud_file_path)[:80]}...{os.path.basename(word_cloud_file_path)[-94:]}'
        plt.savefig(word_cloud_file_path, dpi=600)
    # wordcloud.to_file(word_cloud_file_path)
    if show_plot == 'y':
        plt.show()
    else:
        pause_time = 7
        print(f'Plot has been saved at {word_cloud_file_path}.\nPausing for {pause_time} seconds before proceeding for the next [to avoid pc hang-up]...')
        time.sleep(pause_time)

def wordcloud_from_csv_files_extended(csv_directory_path, csv_file_path, shape_mask_path, word_cloud_file, count_threshold = None, medium_freq_word_threshold = None, minimum_freq_word_threshold = None, max_font_size = 700):
    word_data = defaultdict(lambda: {'values': defaultdict(int), 'total_count': 0, 'constraints': defaultdict(list)})
    all_headers = set()

    for filename in os.listdir(csv_directory_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(csv_directory_path, filename)
            constraints = extract_constraints(filename)
            constraints_key = ' | '.join(constraints)
            all_headers.add(constraints_key)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                word, value = row.iloc[0], row.iloc[1]
                if count_threshold != None and type(count_threshold) == int:
                    if value > count_threshold:  # Only count words with a value greater than 20
                        word_data[word]['total_count'] += 1
                        word_data[word]['values'][constraints_key] += value
                        word_data[word]['constraints'][filename].extend(constraints)
                else:
                    word_data[word]['total_count'] += 1
                    word_data[word]['values'][constraints_key] += value
                    word_data[word]['constraints'][filename].extend(constraints)


    # Save word frequency, filenames, and constraints in CSV file
    save_word_frequencies_to_csv_w_headers(word_data, csv_file_path, sorted(list(all_headers)))


    # Generate wordcloud in shape
    mask_image = Image.open(shape_mask_path).convert('L')
    mask = np.array(mask_image)

    # Generate the word cloud
    print(f'\nPreparing wordcloud image. Please hold on...')
    complete_word_counts = {word: data['total_count'] for word, data in word_data.items()}
    print(f'complete_word_counts: {complete_word_counts}')
    if medium_freq_word_threshold == None or minimum_freq_word_threshold == None:
        wordcloud = WordCloud(
            width=1000,
            height=1000,
            background_color='white',
            mask=mask,
            max_words=500,
            max_font_size=max_font_size,
            scale=10,
        ).generate_from_frequencies(complete_word_counts)

        # Printing each word's font size after generation
        for (word, freq), font_size, position, orientation, color in wordcloud.layout_:
            print(f"Word: '{word}' with frequency {freq} has font size: {font_size}")
    else:
        wordcloud = WordCloud(
            width=1000,
            height=1000,
            background_color='white',
            mask=mask,
            max_words=500,
            max_font_size=max_font_size,
            scale=10,
            color_func=lambda word, font_size, position, orientation, random_state, font_path: custom_color_func(
                word, font_size, position, orientation, random_state, font_path, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, max_font_size=max_font_size
            )
        ).generate_from_frequencies(complete_word_counts)
    # Save the word cloud to a file
    wordcloud.to_file(word_cloud_file)
    # Optionally visualize the wordcloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def wordcloud_from_csv_files(csv_directory_path, csv_file_path, shape_mask_path, word_cloud_file, count_threshold = None, medium_freq_word_threshold = None, minimum_freq_word_threshold = None, max_font_size =700):
    word_cloud_file_path = os.path.join(csv_directory_path, word_cloud_file)
    word_freq_dict = defaultdict(lambda: {'count': 0, 'files': set(), 'constraints': []})

    for filename in os.listdir(csv_directory_path):
        if filename.endswith('.csv'):
            print(f'Checking {filename}')
            filepath = os.path.join(csv_directory_path, filename)
            constraints = extract_constraints(filename)
            # df = pd.read_csv(filepath, usecols=[0])
            # words = df.iloc[:, 0].tolist()
            # for word in words:
            #     word_freq_dict[word]['count'] += 1
            #     word_freq_dict[word]['files'].add(filename)
            #     word_freq_dict[word]['constraints'].append(constraints)  # Append the list of constraints

            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                word, value = row.iloc[0], row.iloc[1]
                # print(f'Terms found in the file: {word}')
                if count_threshold != None and type(count_threshold) == int:
                    if value > count_threshold:  # Only count words with a value greater than 20
                        word_freq_dict[word]['count'] += 1
                        word_freq_dict[word]['files'].add(filename)
                        word_freq_dict[word]['constraints'].append(constraints)  # Append the list of constraints
                else:
                    word_freq_dict[word]['count'] += 1
                    word_freq_dict[word]['files'].add(filename)
                    word_freq_dict[word]['constraints'].append(constraints)  # Append the list of constraints

    # Generate the word cloud
    complete_word_counts = {word: data['count'] for word, data in word_freq_dict.items()}
    print(f'word_freq_dict: {word_freq_dict}\n complete_word_counts: {complete_word_counts}')

    # Save word frequency and filenames in CSV file
    save_word_frequencies_to_csv_w_constraints(word_freq_dict, csv_file_path)

    # Generate wordcloud in shape
    mask_image = Image.open(shape_mask_path).convert('L')
    mask = np.array(mask_image)

    print(f'\nPreparing wordcloud image. Please hold on...')
    if medium_freq_word_threshold == None or minimum_freq_word_threshold == None:
        wordcloud = WordCloud(
            width=1000,
            height=1000,
            background_color='white',
            mask=mask,
            max_words=500,
            max_font_size=max_font_size,
            scale=10,
        ).generate_from_frequencies(complete_word_counts)

        # Printing each word's font size after generation
        for (word, freq), font_size, position, orientation, color in wordcloud.layout_:
            print(f"Word: '{word}' with frequency {freq} has font size: {font_size}")
    else:
        wordcloud = WordCloud(
            width=1000,
            height=1000,
            background_color='white',
            mask=mask,
            max_words=500,
            max_font_size=max_font_size,
            scale=10,
            color_func=lambda word, font_size, position, orientation, random_state, font_path: custom_color_func(
                word, font_size, position, orientation, random_state, font_path,
                medium_freq_word_threshold=medium_freq_word_threshold,
                minimum_freq_word_threshold=minimum_freq_word_threshold, max_font_size=max_font_size
            )
        ).generate_from_frequencies(complete_word_counts)
    plt.figure(figsize=(100, 50))
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(word_cloud_file_path)
    plt.show()

def clean_ris_file(input_file_path, output_file_path):
    # Define printable characters
    printable = set(string.printable)

    # Open the input file and output file
    with open(input_file_path, "r", encoding="utf-8") as file, \
            open(output_file_path, "w", encoding="utf-8") as outfile:
        for line in file:
            # Remove non-printable characters
            cleaned_line = ''.join(filter(lambda x: x in printable, line))
            # Write the cleaned line to the output file
            outfile.write(cleaned_line)

def no_pattern_ID_search(max_font_size: int = 700,
                         shape_mask_path: str | None = None,
                         args: argparse.Namespace | None = None):
    # ------------------------------------------------------------------
    # CLI MODE: if an argparse.Namespace is provided, skip interactive
    # prompts and read settings directly from `args`.
    # ------------------------------------------------------------------
    if args is not None:
        print("[CLI] Running in non‑interactive mode…")
        exclude_words = (parse_user_input_simple_list(args.exclude_words)
                         if getattr(args, 'exclude_words', None) else None)
        exclude_words_phrase = "y" if exclude_words else "n"

        inclusion_terms_file = (
            args.inclusion_terms_file or
            f"{base_dir}/Data/assets/sorting parameters/"
            f"Keywords for screening MADS-box gene IDs_with_synonymous.txt"
        )

        include_only = []
        try:
            with open(inclusion_terms_file, "r") as fp:
                for line in fp:
                    clean = line.strip()
                    if clean:
                        include_only.append([tok.strip() for tok in clean.split(",")])
        except Exception:
            include_only = None

        wordcase   = "sensitive" if args.case_sensitive else "insensitive"
        medium_freq_word_threshold  = args.medium_freq_threshold
        minimum_freq_word_threshold = args.minimum_freq_word_threshold if hasattr(args, 'minimum_freq_word_threshold') else args.minimum_freq_threshold
        plot_show = "y" if args.show_plot else "n"

        constraints = None
        constraints_listoflists = "n"
        if args.constraints:
            if args.constraints_listoflists:
                constraints = parse_user_input_list_of_lists(args.constraints)
                constraints_listoflists = "y"
            else:
                constraints = parse_user_input_simple_list(args.constraints)

        meta_constraints  = (
            parse_user_input_list_of_lists(args.meta_constraints)
            if args.meta_constraints else None
        )
        mutual_inclusivity = "y" if args.mutual_inclusivity else None

        sorting_parameter = os.path.basename(inclusion_terms_file).split(".")[0]
        out_dir = os.path.join(base_dir, "Data", "outputs")
        os.makedirs(out_dir, exist_ok=True)  # create it if missing
        ref_file = (
            f"{base_dir}/Data/assets/References used/"
            "Combo MADS-related studies_wo genome-wide_wo reviews_wo evolutionary studies_nascent.ris.txt"
        )
        references = read_ris_file(ref_file)

        csv_file        = f"Tair-derived reference_{sorting_parameter}_cli_run_word-frequency.csv"
        word_cloud_file = f"Tair-derived reference_{sorting_parameter}_cli_run.png"

        create_word_cloud(
            references,
            shape_mask_path,
            out_dir,
            csv_file,
            word_cloud_file,
            exclude_words_phrase=exclude_words_phrase,
            constraints_listoflists=constraints_listoflists,
            medium_freq_word_threshold=medium_freq_word_threshold,
            minimum_freq_word_threshold=minimum_freq_word_threshold,
            constraints=constraints,
            exclude_words=exclude_words,
            include_only=include_only,
            mutual_inclusivity=mutual_inclusivity,
            wordcase=wordcase,
            patterns=None,
            show_plot=plot_show,
            max_font_size=max_font_size,
        )
        print(f"[CLI] Word‑cloud saved to: {word_cloud_file}")
        return

    print(f'Carrying out no_pattern_ID_search()')
    shape_mask_path = input("\nPlease insert the path to a PNG mask image for wordcloud shape (or press enter for no mask): ")
    if not shape_mask_path:
        shape_mask_path = f"{base_dir}/Data/assets/mask images/Circle-for-wordcloud_.png"
        print(f"Using circle mask as default from {shape_mask_path}")
    out_dir = f'{base_dir}/Data/outputs'
    ref_file_path = input(f"\nPlease insert path of a file that contains study details including title and\n"
                          f"abstract among others in a proper ris format (can either be .ris or .txt)\n"
                          f"Please do not quote the path while entering. For example run, just press enter: ")
    if ref_file_path == "":
        print(f"Using references at {base_dir}/Data/assets/References used/"
              f"Combo MADS-related studies_wo genome-wide_wo reviews_wo evolutionary studies_nascent.ris.txt")
        ref_file_path =f"{base_dir}/Data/assets/References used/" \
                       f"Combo MADS-related studies_wo genome-wide_wo reviews_wo evolutionary studies_nascent.ris.txt"
    # mock_ref_file_path = f'{base_dir}}Data/assets/References_used/test refs.ris.txt'
    references = read_ris_file(ref_file_path)
    patterns = None

    constraints = input('\nPlease enter constraints in list. These are the keywords for which the script will screen the \n'
                        'literatures. (If you do not wish to do so, simply press enter): ')

    # print(f'type(constraints): {type(constraints)}')
    if len(constraints) > 1:
        constraints_listoflists = input('\nIs the constraints list of lists (enter "n" otherwise)? (y/n): ')
        if lower(constraints_listoflists) == 'n':
            try:
                constraints = parse_user_input_simple_list(constraints)
            except Exception as e:
                print(
                    f'No valid constraint input found: {e}\nproceeding without constraints defined. (Restart if you wish to make corrections.)')
                constraints = None
        elif lower(constraints_listoflists) == 'y':
            try:
                constraints = parse_user_input_list_of_lists(constraints)
            except Exception as e:
                print(
                    f'No valid constraint input found: {e}\nproceeding without constraints defined. (Restart if you wish to make corrections.)')
                constraints = None
                constraints_listoflists = 'n'
    else:
        ex_run = input(f"\nIs this an example run? (y/n): ")
        if ex_run == "y":
            print(f"Using fruits-associated meta constraints.")
            meta_constraints = [['fruit set'], ['fruit ripening'], ['fruit size'], ['fruit shape'],
                           ['fruit weight', 'fruit yield'], ['fruit development', 'fruit growth'],
                           ['fruit senescence']]
            print(f"\n Taking default meta constraint list:\n{meta_constraints}")
            # constraints = None
            constraints_listoflists = 'y'
        else:
            constraints = None
            constraints_listoflists = 'n'
    # constraints = None
    # Use meta_constraints (list of constraints lists) instead to prepare multiple wordclouds sequentially
    # meta_constraints = [['maternal'], ['paternal'], ['imprinting', 'imprinted', 'imprint'], ['xylem'], ['seed coat'], ['integument', 'integuments'], ['phloem'], ['root patterning', 'root morphogenesis'], ['stomata'], ['quiescent center'], ['leaf', 'leaves'], ['shoot', 'stem'], ['SAM', 'apical meristem'], ['root'], ['root cap'], ['branching'], ['flowering', 'bolting'], ['flower'], ['fruit'], ['fruit set'], ['transition'], ['pollen'], ['sperm cell'], ['egg cell'], ['central cell'], ['embryo'], ['endosperm'], ['somatic embryogenesis'], ['embryogenesis'], ['seed'], ['seed set'], ['germination'], ['juvenile', 'young'], ['mature', 'adult'], ['cell cycle'], ['root hair'], ['reproduction', 'reproductive'], ['pollen tube'],['fruit size'], ['seed size'], ['nucellus'], ['photoperiod'], ['photosynthesis'], ['life cycle'], ['dormant', 'dormancy'], ['miR156'], ['floral meristem']]
    # meta_constraints = [['maternal'], ['paternal'], ['imprinting', 'imprinted', 'imprint']]
    # meta_constraints = [['root', 'roots'], ['stem', 'shoot'], ['leaf', 'leaves'], ['flower', 'floral'], ['apical meristem', 'SAM'], ['fruit', 'fruits'], ['seed', 'seeds']] # major organs/tissues
    # meta_constraints = [['tendril'], ['trichome'], ['xylem'], ['phloem'], ['lignin']] # sub-tissues
    # meta_constraints = [['photosynthesis'], ['photoperiod'], ['vernalization'], ['clock', 'circadian clock'], ['senescence']] # physiological attributes
    # meta_constraints = [['auxin'], ['cytokinin'], ['gibberellin'], ['abscisic acid'], ['salicylic acid'], ['jasmonic acid'], ['ethylene']] # hormones
    # meta_constraints = [['leaf size'], ['leaf shape'], ['leaf senescence'], ['leaf morphology'], ['chlorophyll'], ['stomata']] # leaf characteristics
    # meta_constraints = [['ovule', 'female gametophyte'], ['egg cell'], ['central cell'], ['synergid cell'], ['antipodal cell'], ['integument']] # ovule parameters
    # meta_constraints = [['pollen', 'male gametophyte'], ['sperm cell'], ['pollen tube'], ['anther']] # male gamate parameters
    # meta_constraints = [['seed coat'], ['seed development'], ['seed set', 'seed-set'], ['seed size'], ['seed shape'], ['embryo'], ['endosperm'], ['nucellus', 'hypostase'], ['peripheral endosperm'], ['chalazal endosperm'], ['syncitial endosperm'], ['seed abortion', 'seed abort'], ['nucellar embryo']] # seed parameters
    # meta_constraints = [['root hair'], ['lateral root'], ['quiescent center'], ['root cap', 'root tip'], ['root length'], ['root development', 'root growth']] # root parameters
    # meta_constraints = [['fruit set'], ['fruit ripening'], ['fruit size'], ['fruit shape'], ['fruit weight', 'fruit yield'], ['fruit development', 'fruit growth'], ['fruit senescence']] # fruit parameters
    # meta_constraints = [['cell cycle'], ['organogenesis', 'patterning', 'redifferentiation'], ['dedifferentiation', 'callus', 'callogenesis'], ['embryogenesis'], ['somatic embryogenesis'], ['regeneration']] # cellular/tissue-level attributes
    # meta_constraints = [['branch', 'branching', 'tiller', 'tillering'], ['node', 'inter-node', 'internode'], ['axil', 'axillary'], ['height', 'stature', 'tall', 'dwarf', 'stunt', 'stunted']] # shoot parameters
    # meta_constraints = [['dormant', 'dormancy'], ['germination'], ['phase change', ' phase transition'], ['mature', 'adult', 'reproductive growth', 'reproductive development'], ['juvenile', 'young', 'vegetative growth', 'vegetative development'], ['pollination'], ['fertilization']] # growth parameters
    # meta_constraints = [['imprinting', 'imprinted'], ['maternal'], ['paternal'], ['hybrid vigor']] # genetic attributes
    # meta_constraints = [['MAPK'], ['miR156', 'miR172']] # Regulatory systems/pathways
    # meta_constraints = [['drought', 'waterlogging'], ['heat'], ['light'], ['disease', 'pathogen'], ['salt', 'salinity'], ['osmotic'], ['wound'], ['ROS', 'oxidative'], ['tolerance', 'resistance', 'tolerant', 'resistant'], ['nutrient deficiency'], ['wind'], ['lodging']] #stresses
    # meta_constraints = [['root', 'shoot'], ['root', 'shoot', 'leaf'], ['root', 'SAM'], ['root', 'flower']] # root-to-others (mutuinclu constraints)
    # meta_constraints = [['nitrate', 'flower'], ['stress', 'flower'], ['ABA', 'flower'], ['auxin', 'flower']] # factors-to-flower (mutuinclu constraints)
    # meta_constraints = [['auxin', 'ovule'], ['nitric oxide', 'ovule'], ['cytokinin', 'ovule'], ['gibberellin', 'ovule']] # factors-to-ovule (mutuinclu constraints)
    # meta_constraints = None

    # # define exclusion terms
    # exclude_words = ['word1', 'word2', 'word3']  # Words to exclude (or None to disable)
    exclude_words = input("\nPlease enter exclusion terms in list form ['shoot meristem', 'shoot meristems', "
                          "'shoot apex', 'shoot apexes', 'shoot apices', 'stem cell'] for shoot. "
                          "(If you do not wish to do so, simply press enter): ")
    if exclude_words != "":
        try:
            exclude_words = parse_user_input_simple_list(exclude_words)
            print(f'exclude_words: {exclude_words}')
            exclude_words_phrase = 'y'
        except Exception as e:
            print(
                f'No valid constraint input found: {e}\nproceeding without constraints defined. (Restart if you wish to make corrections.)')
            exclude_words_phrase = 'n'
            exclude_words = None
    else:
        exclude_words_phrase = 'n'
        exclude_words = None
    # include_only = ['specific', 'words', 'here']  # Words to include (or None to disable)
    include_only = None
    inclusion_terms_file = input(f"\nPlease insert path of a text file containing lists of terms relavant to your study\n"
                                 f"objective (e.g., flowering, anther, pollen, angiosperm, etc.). The downstream assessment\n"
                                 f"will essentially base on the studies containing these terms. Just hit enter if "
                                 f"you're doing an example run: ")
    if inclusion_terms_file == "":
        print(f"\nNo list provided taking example list from {base_dir}/Data/assets/sorting parameters/Keywords\nfor "
              f"screening MADS-box gene IDs_with_synonymous.txt")
        inclusion_terms_file = f"{base_dir}/Data/assets/sorting parameters/" \
                               f"Keywords for screening MADS-box gene IDs_with_synonymous.txt"
    mutual_inclusivity = None # Setting default-case parameter

    sorting_parameter = os.path.basename(inclusion_terms_file).split(".")[0]
    try:
        include_only = []
        with open(inclusion_terms_file, 'r') as f:
            for line in f:
                # include_only.append(line.strip())
                # Strip whitespace and check if the line is not empty
                clean_line = line.strip()
                if clean_line:
                    # Split the line on commas and strip spaces from each term
                    terms = [term.strip() for term in clean_line.split(',')]
                    include_only.append(terms)
            print(f'include_only = {include_only}')
    except Exception as e:
        print(f'No fixed inclusion terms [{e}]. Taking that user is likely using patterns instead.')

    wordcaseQ = input(f'Case-sensitive terms? (y/n): ')
    if wordcaseQ.lower() == 'y':
        wordcase = 'sensitive'
    else:
        wordcase = 'insensitive'

    # Step to assign color threshold to the terms based on their frequency/font-size
    opt1 = input('\nWould you like to customize color range based on word frequency?\n'
                 '[Note: you need to run a test run first notice the "font-to-max_font ratio" in the printed output'
                 'and find the suitable value for the medium and minimum threshold.] (y/n): ')
    if opt1.lower() == 'y':
        while True:
            try:
                medium_freq_word_threshold = input('\nPlease provide the medium word frequency threshold (float/iinteger): ')
                try:
                    medium_freq_word_threshold = int(medium_freq_word_threshold)
                except ValueError:
                    medium_freq_word_threshold = float(medium_freq_word_threshold)
                break
            except Exception as e:
                print(f'Error detected in the input: {e}.\nPlease try again and provide valid numerical input.')
        while True:
            try:
                minimum_freq_word_threshold = input(
                    'Please provide the minimum word frequency threshold (float/iinteger): ')
                try:
                    minimum_freq_word_threshold = int(minimum_freq_word_threshold)
                except ValueError:
                    minimum_freq_word_threshold = float(minimum_freq_word_threshold)
                break
            except Exception as e:
                print(f'Error occurred: {e}.\nPlease try again and provide valid numerical input.')
    else:
        medium_freq_word_threshold = None
        minimum_freq_word_threshold = None

    plot_show = input('\nDisplay the wordcloud plot at the end? (y/n): ')

    try:
        if meta_constraints or meta_constraints != None:
            mutual_inclusivity = input(f'Mutually inclusive constraints in case there are more than one terms for constraints? (y/n): ')
            for constraints in meta_constraints:
                if mutual_inclusivity == 'n':
                    csv_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints).replace("/", "~")}_exclusion_terms-{exclude_words}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                    word_cloud_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints).replace("/", "~")}_exclusion_terms-{exclude_words}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
                else:
                    csv_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints).replace("/", "~")}_exclusion_terms-{exclude_words}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                    word_cloud_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints).replace("/", "~")}_exclusion_terms-{exclude_words}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
                try:
                    print(f'\nPreparing Gene ID wordcloud with constraints: {constraints}')
                    create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, exclude_words_phrase = exclude_words_phrase, constraints_listoflists = constraints_listoflists, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, constraints=constraints,
                                  exclude_words=exclude_words, include_only=include_only, mutual_inclusivity= mutual_inclusivity,wordcase=wordcase, patterns = patterns, show_plot = plot_show, max_font_size = max_font_size)
                except Exception as e:
                    print(f'Error occured: {e}')

    except Exception as e:
        print(f'No meta_constraints defined: {e}. Running without it.')
        print(f'constraints: {constraints}\nconstraints length: {len(constraints) if constraints is not None else constraints}')
        if constraints not in (None, 'n') and len(constraints) > 1:
            mutual_inclusivity = input(f'Mutually inclusive constraints {constraints}? (y/n): ')
            mutual_inclusivity = mutual_inclusivity.lower()
        elif constraints not in (None, 'n') and len(constraints) == 1:
            mutual_inclusivity = None
        else:
            mutual_inclusivity = None
        if mutual_inclusivity in (None, 'n'):
            csv_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints)}_exclusion_terms-{exclude_words}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
            word_cloud_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints)}_exclusion_terms-{exclude_words}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
        else:
            csv_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints)}_exclusion_terms-{exclude_words}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
            word_cloud_file = f'Tair-derived reference_{sorting_parameter}_constrain-{str(constraints)}_exclusion_terms-{exclude_words}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'

        if patterns:
            try:
                print(f'Sorting terms/keywords/genes for the wildcard.')
                create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, exclude_words_phrase = exclude_words_phrase, constraints_listoflists = constraints_listoflists, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, constraints = constraints, exclude_words= exclude_words, include_only = include_only, wordcase = wordcase, patterns = patterns, show_plot=plot_show,  max_font_size = max_font_size)
            except Exception as e:
                print(f"Error encountered. Likely no term/keyword was returned for wordcloud to generate:\n{e}")
        else:
            try:
                print(f'Sorting terms/keywords/genes for the single constraints.')
                print(f'mutual_inclusivity = {mutual_inclusivity}')
                create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, exclude_words_phrase = exclude_words_phrase, constraints_listoflists = constraints_listoflists, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, constraints=constraints,
                                  exclude_words=exclude_words, include_only=include_only, mutual_inclusivity= mutual_inclusivity, wordcase = wordcase,
                                  patterns=patterns, show_plot = plot_show, max_font_size = max_font_size)
            except Exception as e:
                print(f"Error encountered. Likely no term/keyword was returned for wordcloud to generate:\n{e}")


# === CLI/Interactive Entrypoint Guard ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WordCloud Generator CLI")
    parser.add_argument('--mode', choices=['cli', 'interactive'], default='interactive',
                        help='Run in CLI or interactive mode')
    parser.add_argument('--shape_mask_path', type=str, default=None, help='Mask image path')
    parser.add_argument('--max_font_size', type=int, default=700, help='Max font size')
    parser.add_argument('--ref_file_path', type=str, help='Path to input RIS/TXT file')
    parser.add_argument('--inclusion_terms_file', type=str, default=None, help='Path to inclusion terms file')
    parser.add_argument('--case_sensitive', action='store_true', help='Enable case sensitive matching')
    parser.add_argument('--constraints_listoflists', action='store_true',
                             help='Treat constraints as list-of-lists')
    parser.add_argument('--exclude_words', type=str, help='Exclude words list')
    parser.add_argument('--constraints', type=str, help='Constraints list or nested list')
    parser.add_argument('--meta_constraints', type=str, help='Meta‐constraints (list-of-lists) as string')
    parser.add_argument('--mutual_inclusivity', action='store_true', help='Use mutual inclusivity')
    parser.add_argument('--medium_freq_threshold', type=float, help='Medium freq threshold')
    parser.add_argument('--minimum_freq_threshold', type=float, help='Min freq threshold')
    parser.add_argument('--show_plot', action='store_true', help='Display plot')
    args = parser.parse_args()

    if args.mode == 'cli':
        no_pattern_ID_search(
            max_font_size=args.max_font_size,
            shape_mask_path=args.shape_mask_path,
            args=args
        )
    else:
        no_pattern_ID_search()

def pattern_search(max_font_size = 700, shape_mask_path = None):
    print(f'Carrying out pattern_search()')
    out_dir = f'{base_dir}'
    # ref_file_path = f'{base_dir}/References used/test refs.ris.txt'  # mock ref file path
    ref_file_path = f'{base_dir}/References used/Combo MADS-related studies_wo genome-wide_wo reviews_wo evolutionary studies_nascent.ris.txt'   # Replace with your .ris file path
    references = read_ris_file(ref_file_path)
    # shape_mask_path = f'{base_dir}/assets/mask images/Circle-for-wordcloud.png'  # Replace with your mask image path. Note, images should be in a format like PNG, where the shape is in black on a white background

    # for pattern search
    inclusion_terms_file = None
    include_only = None
    # constraints = ['seed abortion', 'seed abort'] # None
    # constraints = ['node', 'inter-node', 'internode']#None
    constraints = input('\nPlease enter constraints in list. (If you do not wish to do so, simply press enter): ')
    # print(f'type(constraints): {type(constraints)}')
    if len(constraints) > 1:
        constraints_listoflists = input('\nIs the constraints list of lists (enter "n" otherwise)? (y/n): ')
        if constraints_listoflists == 'n':
            try:
                constraints = parse_user_input_simple_list(constraints)
            except Exception as e:
                print(
                    f'No valid constraint input found: {e}\nproceeding without constraints defined. (Restart if you wish to make corrections.)')
                constraints = None
        elif constraints_listoflists == 'y':
            try:
                constraints = parse_user_input_list_of_lists(constraints)
            except Exception as e:
                print(
                    f'No valid constraint input found: {e}\nproceeding without constraints defined. (Restart if you wish to make corrections.)')
                constraints = None
                constraints_listoflists = 'n'
    else:
        constraints = None
        constraints_listoflists = 'n'
    ex_run = input(f"Is this an example run? (y/n): ")
    if ex_run == "y":
        meta_constraints = [['auxin'], ['cytokinin'], ['gibberellin'], ['abscisic acid'], ['salicylic acid'],
                            ['jasmonic acid'], ['ethylene']]  # hormones
    else:
        meta_constraints = input(f"Please enter list of lists as meta constraints (e.g., [['auxin'], ['cytokinin']]: ")
        meta_constraints = parse_user_input_list_of_lists(meta_constraints)
    print(f"meta_constraints (hormones) has been activated for example run")
    # meta_constraints = None
    # meta_constraints = [['root', 'roots'], ['stem', 'shoot'], ['leaf', 'leaves'], ['flower', 'floral'], ['apical meristem', 'SAM'], ['fruit', 'fruits'], ['seed', 'seeds']] # major plant organs/tissues
    # meta_constraints = [['tendril'], ['trichome'], ['xylem'], ['phloem'], ['lignin']] # sub-tissues
    # meta_constraints = [['vernalization'], ['clock', 'circadian clock'], ['senescence']]  # physiological attributes
    # meta_constraints = [['leaf size'], ['leaf shape'], ['leaf senescence'], ['leaf morphology'], ['chlorophyll'], ['stomata']] # leaf characteristics
    # meta_constraints = [['ovule', 'female gametophyte'], ['egg cell'], ['central cell'], ['synergid cell'], ['antipodal cell'], ['integument']] # ovule parameters
    # meta_constraints = [['pollen', 'male gametophyte'], ['sperm cell'], ['pollen tube'], ['anther']] # male gamate parameters
    # meta_constraints = [['seed coat'], ['seed development'], ['seed set', 'seed-set'], ['seed size'], ['seed shape'], ['embryo'], ['endosperm'], ['nucellus', 'hypostase'], ['peripheral endosperm'], ['chalazal endosperm'], ['syncitial endosperm'], ['seed abortion', 'seed abort'], ['nucellar embryo']] # seed parameters
    # meta_constraints = [['root hair'], ['lateral root'], ['quiescent center'], ['root cap', 'root tip'], ['root length'], ['root development', 'root growth']] # root parameters
    # meta_constraints = [['fruit set'], ['fruit ripening'], ['fruit size'], ['fruit shape'], ['fruit weight', 'fruit yield'], ['fruit development', 'fruit growth'], ['fruit senescence'], ['fruit color'], ['fruit density', 'fruit number']] # fruit parameters
    # meta_constraints = [['cell cycle'], ['organogenesis', 'patterning', 'redifferentiation'], ['dedifferentiation', 'callus', 'callogenesis'], ['embryogenesis'], ['somatic embryogenesis'], ['regeneration']] # cellular/tissue-level attributes
    # meta_constraints = [['branch', 'branching', 'tiller', 'tillering'], ['node', 'inter-node'], ['axil', 'axillary'], ['height', 'stature', 'tall', 'dwarf', 'stunt', 'stunted']] # shoot parameters
    # meta_constraints = [['dormant', 'dormancy'], ['germination'], ['phase change', ' phase transition'], ['mature', 'adult', 'reproductive growth', 'reproductive development'], ['juvenile', 'young', 'vegetative growth', 'vegetative development'], ['pollination'], ['fertilization']] # growth parameters
    # meta_constraints = [['imprinting', 'imprinted'], ['maternal'], ['paternal'], ['hybrid vigor']] # genetic attributes
    # meta_constraints = [['MAPK'], ['miR156', 'miR172']] # Regulatory systems/pathways
    # meta_constraints = [['drought', 'waterlogging'], ['heat'], ['light'], ['disease', 'pathogen'], ['salt', 'salinity'], ['osmotic'], ['wound'], ['ROS', 'oxidative'], ['tolerance', 'resistance', 'tolerant', 'resistant'], ['nutrient deficiency'], ['wind'], ['lodging']] #stresses
    # meta_constraints = [['root', 'shoot'], ['root', 'shoot', 'leaf'], ['root', 'SAM'], ['root', 'flower']] # root-to-others (mutuinclu constraints)
    # meta_constraints = [['nitrate', 'flower'], ['stress', 'flower'], ['ABA', 'flower'], ['auxin', 'flower']] # factors-to-flower (mutuinclu constraints)
    # meta_constraints = [['auxin', 'ovule'], ['nitric oxide', 'ovule'], ['cytokinin', 'ovule'], ['gibberellin', 'ovule']] # factors-to-ovule (mutuinclu constraints)
    # MADS_patterns = [r'\b.*MADS.*\b', r'\b.\w*AGL.*\b', r'\b.*DAM.*\b']
    MADS_patterns = [r'\b.*MDP.*', r'\bODDSOC2\b', r'\bTM6\b', r'\bSlMBP3\b', r'\b.FBP.*\b', r'\b\w+AGL.\w*\b', r'\b\w+FUL\w*\b', r'\b\w+FLC\w*\b', r'\b\w+AG.*\b', r'\b\w+AP3\w*\b', r'\b\w+AP1\w*\b', r'\b\w+SEP\w*\b', r'\b\w+SVP\w*\b', r'\b\w+SOC1\w*\b', r'\b\w+TT16\w*\b', r'\b\w+ANR.\w*\b', r'\b.*MADS.*\b', r'\b.RIN.*\b', r'\b.*DAM.*\b', r'\bMdJa.*\b', r'\bDOLL1\b', r'\bTAGL1\b', r'\bNycAG.*\b', r'\bzag.*\b', r'\bPfAG.*\b', r'\bZMM28\b', r'\bVRN1\b']

    MADS_exclude_words = ['PRINCIPAL', 'OE_FUL', 'RNAi_AGL', 'OXEgMADS16', 'FLAG', 'MADSs', 'GAGA', 'FRUITFUL', 'FRUITFULL', 'paleoAP1', 'euAP1', 'gAGL6', 'CALLOSE', 'PHASE', 'STMADS', 'RIN', 'RINs', 'DAM', 'DAMs', 'MADS-mediated', 'MADS-Domain', 'MADS-proteins', 'MADSbox', 'MADSBox', 'MADS-box',
                          'MADS-Box',  'MADS-domain', 'MADS', 'MADS-box-complexes', 'MADS-complexes', 'MADS-box-SHORT', 'MADS-boxes',
                          'MIKCC-MADS-box', 'MIKCCMADS', 'MIKC-MADS', 'MADS-BOX',
                          'DnMADS-box', r'AtAGL.*']  # Words to exclude (or None to disable)
    # MADS_exclude_words = None
    patterns = MADS_patterns
    print(f'term patterns: {patterns}')

    csv_file = f'Tair-derived reference_MADS wildcard_constrain-MADS_patterns_{str(constraints)}_max-font-size_{max_font_size}_word-frequency.csv'
    word_cloud_file = f'Tair-derived reference_MADS wildcard_constrain-MADS_patterns_{str(constraints)}_max-font-size_{max_font_size}.png'

    wordcaseQ = input(f'Case-sensitive terms? (y/n): ')
    if wordcaseQ.lower() == 'y':
        wordcase = 'sensitive'
    else:
        wordcase = 'insensitive'

    # Step to assign color threshold to the terms based on their frequency/font-size
    opt1 = input('\nWould you like to customize color range based on word frequency?\n'
                 '[Note: you need to run a test run first notice the "font-to-max_font ratio" in the printed output'
                 'and find the suitable value for the medium and minimum threshold.] (y/n): ')
    if opt1.lower() == 'y':
        while True:
            try:
                medium_freq_word_threshold = input(
                    'Please provide the medium word frequency threshold (float/iinteger): ')
                try:
                    medium_freq_word_threshold = int(medium_freq_word_threshold)
                except ValueError:
                    medium_freq_word_threshold = float(medium_freq_word_threshold)
                break
            except Exception as e:
                print(f'Error detected in the input: {e}.\nPlease try again and provide valid numerical input.')
        while True:
            try:
                minimum_freq_word_threshold = input(
                    'Please provide the minimum word frequency threshold (float/iinteger): ')
                try:
                    minimum_freq_word_threshold = int(minimum_freq_word_threshold)
                except ValueError:
                    minimum_freq_word_threshold = float(minimum_freq_word_threshold)
                break
            except Exception as e:
                print(f'Error occurred: {e}.\nPlease try again and provide valid numerical input.')
    else:
        medium_freq_word_threshold = None
        minimum_freq_word_threshold = None

    plot_show = input('\nDisplay the wordcloud plot at the end? (y/n): ')

    try:
        if meta_constraints != None and meta_constraints != [] and type(meta_constraints) == list and type(meta_constraints[0]) == list:
            mutual_inclusivity = input(f'Mutually inclusive constraints in case there are more than one terms for constraints? (y/n): ')
            if mutual_inclusivity != 'y' or mutual_inclusivity != 'n' or mutual_inclusivity != None:
                print(f'No valid input found for nutual inclusivity option. Stop the script and re-run it.')
            for constraints in meta_constraints:
                if mutual_inclusivity == None or mutual_inclusivity == 'n':
                    csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                    word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
                elif mutual_inclusivity == 'y':
                    csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                    word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
                try:
                    print(f'\nPreparing Gene ID wordcloud with constraints: {constraints}')
                    create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, constraints_listoflists = constraints_listoflists, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, constraints=constraints,
                                  exclude_words=MADS_exclude_words, include_only=include_only, mutual_inclusivity= mutual_inclusivity, wordcase=wordcase, patterns = MADS_patterns, show_plot = plot_show, max_font_size = max_font_size)
                except Exception as e:
                    print(f'Error occured: {e}')

    except Exception as e:
        print(f'No meta_constraints defined: {e}. Running without it.')
        if constraints not in (None, 'n'):
            if len(constraints) > 1:
                mutual_inclusivity = input(f'Mutually inclusive constraints {constraints}? (y/n): ')
                mutual_inclusivity = mutual_inclusivity.lower()
                if mutual_inclusivity == None or mutual_inclusivity == 'n':
                    csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                    word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
                elif mutual_inclusivity == 'y':
                    csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                    word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_mutuInclu_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
                else:
                    print(f'No valid input found for nutual inclusivity option. Stop the script and re-run it.')
            elif len(constraints) == 1:
                mutual_inclusivity = None
                csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
            elif len(constraints) == 0:
                print(f'No valid constraints found. skipping ahead')
                mutual_inclusivity = None
                csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
                word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'
        else:
            mutual_inclusivity = None
            csv_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_word-frequency.csv'
            word_cloud_file = f'Tair-derived reference_MADS_patterns_constrain-{str(constraints)}_max-font-size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}.png'

        if patterns:
            try:
                print(f'Sorting terms/keywords/genes for the wildcard.')
                create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, constraints_listoflists = constraints_listoflists, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, constraints = constraints, exclude_words= MADS_exclude_words, include_only = include_only, mutual_inclusivity= mutual_inclusivity, wordcase = wordcase, patterns = MADS_patterns, show_plot = plot_show, max_font_size = max_font_size)
            except Exception as e:
                print(f"Error encountered. Likely no term/keyword was returned for wordcloud to generate:\n{e}")
        else:
            try:
                print(f'Sorting terms/keywords/genes for the single constraints.')
                create_word_cloud(references, shape_mask_path, out_dir, csv_file, word_cloud_file, constraints_listoflists = constraints_listoflists, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, constraints=constraints,
                                  exclude_words=MADS_exclude_words, include_only=include_only, mutual_inclusivity= mutual_inclusivity, wordcase=wordcase,
                                  patterns=MADS_patterns, show_plot = plot_show, max_font_size = max_font_size)
            except Exception as e:
                print(f"Error encountered. Likely no term/keyword was returned for wordcloud to generate:\n{e}")

def csv_files_to_wordcloud(max_font_size = 700, shape_mask_path = None):
    # Wordcloud from multiple csv files
    print("running csv_files_to_wordcloud()")
    # shape_mask_path = f'{base}/Data/assets/mask images/Circle-for-wordcloud.png'  # Replace with your mask image path. Note, images should be in a format like PNG, where the shape is in black on a white background
    # shape_mask_path = f'{base}/Data/assets/mask images/Tree-mask__small.png' # tree mask image
    # csv_directory_path = f'{base_dir}/Major tissues'
    # csv_directory_path = input('\nPlease provide the directory path containing target .csv files: ')

    while True:
        csv_directory_path = input('\nPlease provide the directory path containing target .csv files: ')
        if os.path.isdir(csv_directory_path):
            print(f"The path '{csv_directory_path}' is a valid directory path. Moving forward.")
            break
        else:
            print(f"No valid path detected. Please make correction and try again.")

    file_name = os.path.basename(csv_directory_path)
    count_threshold_Q = input(f'Would you like to include the count threshold, below which you do not wish to count? (y/n): ')
    if count_threshold_Q == 'y':
        count_threshold = int(input('\nPlease provide a count threshold below which you do not wish to count (including itself). [Integer value only]: '))
    else:
        print (f'Continuing without count threshold.')
        count_threshold = None
    opt1 = input('\nWould you like to customize color range based on word frequency?\n'
                 '[Note: you need to run a test run first notice the "font-to-max_font ratio" in the printed output'
                 'and find the suitable value for the medium and minimum threshold.] (y/n): ')
    if opt1.lower() == 'y':
        medium_freq_word_threshold = input('\nPlease provide the medium word frequency threshold (float/iinteger): ')
        try:
            medium_freq_word_threshold = int(medium_freq_word_threshold)
        except ValueError:
            try:
                medium_freq_word_threshold = float(medium_freq_word_threshold)
            except:
                print('Error detected in the input. Please provide valid numerical input.')
        minimum_freq_word_threshold = input('\nPlease provide the minimum word frequency threshold (float/iinteger): ')
        try:
            minimum_freq_word_threshold = int(minimum_freq_word_threshold)
        except ValueError:
            try:
                minimum_freq_word_threshold = float(minimum_freq_word_threshold)
            except Exception as e:
                print(f'Error occurred: {e}. Please restart again and provide valid numerical input.')
    else:
        medium_freq_word_threshold = None
        minimum_freq_word_threshold = None
    opt2 = input('\nWould you like to generate concise or extended csv file? (c/e): ')


    if opt2.lower() == 'e':
        print('Preparing a extended csv version of combined csv files,')
        # Path to the directory containing your CSV files
        csv_file_path = f'{os.path.dirname(csv_directory_path)}/{file_name}_max_font_size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_count_threshold-{count_threshold}_extended_version.csv'
        word_cloud_file = f'{os.path.dirname(csv_directory_path)}/{file_name}_max_font_size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_count_threshold-{count_threshold}_extended_version.png'  # 'All csvs to wordcloud_max-font-size_{max_font_size}.png'
        wordcloud_from_csv_files_extended(csv_directory_path, csv_file_path, shape_mask_path, word_cloud_file, count_threshold = count_threshold, medium_freq_word_threshold = medium_freq_word_threshold, minimum_freq_word_threshold = minimum_freq_word_threshold, max_font_size=max_font_size)
    else:
        print('Preparing a concise csv version of combined csv files,')
        csv_file_path = f'{os.path.dirname(csv_directory_path)}/{file_name}_max_font_size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_count_threshold-{count_threshold}_concise_version.csv'
        word_cloud_file = f'{os.path.dirname(csv_directory_path)}/{file_name}_max_font_size_{max_font_size}_med-{medium_freq_word_threshold}_min-{minimum_freq_word_threshold}_count_threshold-{count_threshold}_concise_version.png'  # 'All csvs to wordcloud_max-font-size_{max_font_size}.png'
        wordcloud_from_csv_files(csv_directory_path, csv_file_path, shape_mask_path, word_cloud_file, count_threshold = count_threshold, max_font_size = max_font_size)

def main():
    """Entry point when executed as a script (non-interactive)."""
    args = _parse_cli_args()


    max_font_size = args.max_font_size
    shape_mask_path = (
        args.shape_mask_path
        or f"{base_dir}/Data/assets/mask images/Circle-for-wordcloud_.png"
    )

    no_pattern_ID_search(
        max_font_size=max_font_size,
        shape_mask_path=shape_mask_path,
        args=args,
    )


    # shape_mask_path = f'{base_dir}/mask images/Circle-for-wordcloud_.png' # general shape mask path
    # shape_mask_path = f'{base_dir}/mask images/Root-mask.png' # root_mask_path
    # shape_mask_path = f'{base_dir}/mask images/shoot-mask.png # shoot_mask_path'
    # shape_mask_path = f'{base_dir}/mask images/leaf-mask_horiz_small.png' # leaf_mask_path
    # shape_mask_path = f'{base_dir}/mask images/flower-mask_dandelion.png # flower_mask_path'
    # shape_mask_path = f'{base_dir}/mask images/Fruit-mask_mango.png' # fruit_mask_path
    # shape_mask_path = f'{base_dir}/mask images/seed-mask.png' # seed_mask_path
    # shape_mask_path = f'{base_dir}/mask images/SAM-mask.png' # SAM_mask_path
    # shape_mask_path = f'{base_dir}/mask images/ovule-mask.png' # ovule_mask_path
    # shape_mask_path = f'{base_dir}/mask images/germination-mask.png' # germination_mask_path
    # shape_mask_path = f'{base_dir}/mask images/Tree-mask__small.png' # tree_mask_path
    # shape_mask_path = f'{base_dir}/mask images/pollen-mask_wavy.png' # pollen_mask_wavy_path
    # shape_mask_path = f'{base_dir}/mask images/auxin_arrow_mask.png' # auxin_mask_path
    # shape_mask_path = f'{base_dir}/mask images/Cytokinin_star_mask.png' # cytokinin_mask_path
    # shape_mask_path = f'{base_dir}/mask images/gibberellin_sunburst_mask.png' # gibberellin_mask_path
    # shape_mask_path = f'{base_dir}/mask images/abscissic-acid_hourglass_mask.png' # abscissic_acid_mask_path
    # shape_mask_path = f'{base_dir}/mask images/ethylene_bubble_mask.png' # ethylene_mask_path
    # shape_mask_path = f'{base_dir}/mask images/nutrients_molecule_mask.png' # nutrient_mask_path
    # shape_mask_path = f'{base_dir}/mask images/resistance_tolerance_susceptibility_mask.png' # resistance/tolerance/susceptibility_mask_path
    # shape_mask_path = f'{base_dir}/mask images/light_mask.png' # light_mask_path
    # shape_mask_path = f'{base_dir}/mask images/heat_mask.png' # heat_mask_path
    # shape_mask_path = f'{base_dir}/mask images/salt_mask.png' # salt_mask_path
    # shape_mask_path = f'{base_dir}/mask images/osmosis_mask.png' # osmotic_pressure_path
    # ['shoot', 'stem']
    # exclude_words (for shoot): ['shoot meristem', 'shoot meristems', 'shoot apex', 'shoot apexes', 'shoot apices', 'stem cell']
    # ['flower', 'flowers', 'floral']
    # ['ovule', 'ovules', 'female gametophyte', 'female gametophytes']
    # ['pollen', 'pollens', 'male gametophyte', 'male gametophytes']
    # ['pollen tube guidance']
    # ['seed', 'seeds', 'grain', 'kernel']
    # ['seed plant', 'seed plants', 'seed-plant', 'seed-plants']
    # ['fruit', 'fruits']
    # [['seed', 'seeds'], ['germination', 'sprouting']]
    # ['gibberellin', 'GA3', 'gibberellins']
    # ['abscisic acid', 'ABA']
    # ['nutrient', 'nutrients', 'nitrogen', 'phosphorus', 'potassium']
    # light_exclusion_terms = ['shed light', 'sheds light', 'shedding light', 'in light of', 'in the light of', 'light weight', 'brings to light', 'bring to light', 'brought to light', 'bringing to light', 'light pink', 'light red', 'light green', 'light blue']
    no_pattern_ID_search(max_font_size = max_font_size, shape_mask_path = shape_mask_path)
    # pattern_search(max_font_size = max_font_size, shape_mask_path = shape_mask_path)
    # csv_files_to_wordcloud(max_font_size = max_font_size, shape_mask_path = shape_mask_path)


if __name__ == "__main__":
    main()

