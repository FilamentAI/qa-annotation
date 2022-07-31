import uuid
import glob
import json
import os
import hashlib
import collections
import argparse
import time
import random
import logging
import sys
from typing import List, Dict, Tuple, Union, Optional

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger('Annotation Tool')
logging.basicConfig(level='INFO', format='%(levelname)s: %(message)s (Line %(lineno)d)', stream=sys.stdout)

INPUT_DATA_FOLDER           = os.environ.get('FQA_INPUT_DATA_FOLDER', os.path.join('data', 'generated_questions'))
PROFILES_FOLDER             = os.environ.get('FQA_ANNOTATION_DATA_FOLDER', os.path.join('data', 'qatool_annotations', 'full'))
PRELIMINARY_PROFILES_FOLDER = os.environ.get('FQA_PRELIMINARY_ANNOTATION_DATA_FOLDER', os.path.join('data', 'qatool_annotations', 'preliminary'))

os.makedirs(INPUT_DATA_FOLDER, exist_ok=True)
os.makedirs(PROFILES_FOLDER, exist_ok=True)
os.makedirs(PRELIMINARY_PROFILES_FOLDER, exist_ok=True)

def shuffle_qa_data(username: str, qa_data: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Shuffles the QA data and returns a shuffled version, using the username as a random seed for reproducibility.

    Contexts are still "grouped" after shuffling, such that all QA pairs with the same context are contiguous,
        but their order within that grouping will be shuffled as well as the order of contexts.

    Note that if you're previously set the random seed, you'll need to set it again after calling this.

    :param username: The username to use as a random seed.

    :param qa_data: The data to shuffle. Should be a list of (context, question, answer) triples. Not modified.

    :return: A shuffled copy of the QA data.
    """

    return qa_data # TODO Delete this line.

    random.seed(username)

    sub_groups = {}

    for context, question, answer in qa_data:
        if context not in sub_groups:
            sub_groups[context] = []

        sub_groups[context].append( (question, answer) )

    shuffled_sub_lists = []
    for context, qa_list in sub_groups.items():
        shuffled_sub = random.sample(qa_list, k=len(qa_list))

        shuffled_sub_lists.append([ (context, question, answer) for question, answer in shuffled_sub ])

    shuffled_data = random.sample(shuffled_sub_lists, k=len(shuffled_sub_lists))

    flat_data = []

    for sub_list in shuffled_data:
        flat_data.extend(sub_list)

    return flat_data

def get_profile_dir(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's profile directory.

    Does not ensure that the dir exists.

    :param username: The username of the user whose data will be contained there.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    if subset_index is not None:
        username = os.path.join(str(subset_index), username)

    if preliminary:
        return os.path.join(PRELIMINARY_PROFILES_FOLDER, username)
    else:
        return os.path.join(PROFILES_FOLDER, username)

def get_profile_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's profile.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'profile.json')

def get_squad_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's SQuAD data.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'profile.squad')

def get_unsuitable_questions_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's "unsuitable questions" file.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'unsuitable_questions.json')

def get_unnatural_questions_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's "unnatural texts" file.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'unnatural_texts.json')

def get_incorrect_qa_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's "incorrect QA pairs" file.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'incorrect.json')

def get_times_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's "times" file.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'times.json')

def get_notes_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Generates the path to a user's "notes" file.

    Does not ensure that the file exists.

    :param username: The username of the user whose data populated the file.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return: The relative filepath.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'notes.json')

def get_data_filepath(preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Gets the filepath to the data for them to annotate.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. If set, the index of the subset of the data we're running with.

    :return The filepath.
    """

    if preliminary:
        return os.path.join(INPUT_DATA_FOLDER, 'preliminary_generated_data.json')
    else:
        if subset_index is not None:
            return os.path.join(INPUT_DATA_FOLDER, f'subset_{subset_index}_generated_data.json')
        else:
            return os.path.join(INPUT_DATA_FOLDER, 'generated_data.json')

def get_completeness_marker_filepath(username: str, preliminary: bool, subset_index: Optional[int] = None) -> str:
    """
    Gets the filepath to the file that marks a user as having completed the annotation.

    :param username: The username of the user who completed the study.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.
    """

    return os.path.join(get_profile_dir(username, preliminary, subset_index), 'complete')

def get_question_instructions_examples() -> str:
    """
    Get the instructions and examples for judging questions to provide users during the calibration process.

    :return: A HTML/Markdown-compatible string with the instructions and examples.
    """

    return '''A <span style="color: green">**suitable**</span> question will be answerable based on the document without requiring external information, and should be relevant to the document.

As well as being suitable, the question should read naturally: Its meaning should be clear and it should read like fluent English. However, it doesn't have to be perfectly grammatical. 

**Document**
> "'Widget Inc. achieved profits of £50'000 in the second quarter of 2018', reported CEO John McMillan this week, as tech industry stock prices rose across the board"

<hr/>

<span style="color: green">Valid example questions</span>:  
* "What company achieved profits of £50'000 in the second quarter of 2018?" 
* "Who is the CEO of Widget Inc?"

These questions can be considered suitable without modification. 
Note that we don't consider the answer, because as long as a question is answerable from the document, the answer itself doesn't matter. 

For instance, the initial answer may be wrong and need corrections, but as long as **you** can determine the correct answer, the question itself is fine. 

<hr/>

<span style="color: orange">Suitable but non-natural questions, and corrections</span>: 

* "What is the company name that achieved £50'000 in profit in the second quarter of 2018?" -> "Which company achieved profits of £50'000 in the second quarter of 2018?" 
* "Which number quarter of 2018 were the profits from?" -> "Which quarter of 2018 did Widget Inc. earn the profits in?"

These questions don't read naturally in their original forms, though their meanings can be understood. 

They should be marked as **unnatural** and corrected via the provided textbox whilst preserving the overall meaning, but they should **not** marked as unsuitable.

<hr/>

<span style="color: red">**Unsuitable questions**</span>
* "Who is the Chief Financial Officer of Widget Inc?" -> This is not stated in the document, and so the question is impossible to answer.
* "What fires can be started?" -> As well as not being stated in the document, this question makes no sense as a product of the document, as it is completely irrelevant.

These questions should be marked as unsuitable. 
'''

def get_answer_instructions_examples() -> str:
    """
    Get the instructions and examples for judging answers to provide users during the calibration process.

    :return: A HTML/Markdown-compatible string with the instructions and examples.
    """

    return '''A <span style="color: green">**suitable**</span> answer will read naturally and correctly answer the question based on the information in the document.

Answers must be a case-sensitive snippet of the document.

This naturalness should be relative to the document: It doesn't need to have perfect grammar, but it should read easily and naturally, without extra effort to work out the meaning.

As well as reading naturally, an answer may be judged to be "adequate" - if it answers the question correctly when paired with the context (but may have missing or unnecessary detail), and "precise and correct" which additionally means that there is no missing or unnecessary detail from the context.

Answers may be marked as any of these within the tool. Any precise-and-correct answer will necessarily also be adequate, though it might not read naturally. 

**Document**
> "'Widget Inc. achieved profits of £50'000 in the second quarter of 2018', reported CEO John McMillan this week, as tech industry stock prices rose across the board"

**Question**
> "When did Widget Inc. achieve profits of £50'000?"

<hr/>

<span style="color: green">Precise and correct example answer</span>

> in the second quarter of 2018

This answer is precise and factually correct. There's no missing information or extra.

<hr/>

<span style="color: orange">Adequate, but imprecise answers</span>

> 2018'

This answer is **correct**, and would be fine when paired with the document, but it is also **imprecise** as we can provide more information about *when* in 2018 they were achieved.

Thus, it is adequate, but imprecise. The extra apostrophe is unnecessary, but does not affect readability so it can be ignored here.

> in the second quarter of 2018', reported

This answer is **correct**, but it has some unnecessary text at the end ("reported") and can be made more precise by removing that. Thus, it is adequate but not precise.

<hr/>

<span style="color: red">**Incorrect**</span>:

> this week

This answer reads naturally, but it is incorrect and should be marked as such. It should then be corrected using the provided text box.
'''

def scroll_to_top():
    """
    Produces a component which scrolls the window to the top when rendered.
    """

    # Components put it inside an iframe so we first escape that.
    # We need to add the time comment to force the component to be re-rendered, since the script itself doesn't change.
    components.html(f'''<!--{time.time()}-->
<script language="javascript">
    window.parent.document.querySelector("section.main").scrollTo(0, 0)
</script>''', width=0, height=0)

def kept_pairs_to_output(kept_pairs: List[Dict[str, str]]) -> Tuple[Dict[str, Union[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]], str]], List[str], List[Tuple[str, str]]]:
    """
    Converts data in our kept_pairs format to the SQuAD V2 format, filtering out unnatural/incorrect questions and answers into their own dataset.

    All questions are currently assumed to be answerable, as the tool does not presently refer to impossible questions.

    :param kept_pairs: The kept pairs to convert.

    :return: A tuple containing:
        The valid QA pairs (incl. user-submissions) in SQuAD format.
        A list of unnatural questions and answers
        A list of pairs of questions and their incorrect answers
    """

    unnatural = []
    incorrect = []
    context_to_question_answers = {}

    for kept_pair in kept_pairs:
        context = kept_pair['Context']
        question = kept_pair['Question']
        answer = kept_pair['Answer']

        user_question = kept_pair['User Query']
        user_answer = kept_pair['User Answer']

        if context not in context_to_question_answers:
            context_to_question_answers[context] = {}

        pairs_to_add = [ (user_question, user_answer) ]

        if kept_pair['Original Question Naturalness'] and \
                kept_pair['Original Answer Naturalness'] and \
                (kept_pair['Original Answer Adequacy'] or kept_pair['Original Answer Correctness']):
            pairs_to_add.append( (question, answer) )
        else:
            if not kept_pair['Original Question Naturalness']:
                unnatural.append(question)

            if not kept_pair['Original Answer Naturalness']:
                unnatural.append(answer)

            if not kept_pair['Original Answer Correctness'] and not kept_pair['Original Answer Adequacy']:
                incorrect.append( (question, answer) )

        for q, a in pairs_to_add:
            if q not in context_to_question_answers[context]:
                context_to_question_answers[context][q] = []

            try:
                context_to_question_answers[context][q].append({'text': a, 'answer_start': context.index(a)})
            except ValueError as e:
                logger.exception(e)
                logger.error(st.session_state)
                logger.error([context, q, a])

    squad = {
        'data': [
            {
                'title': 'Streamlit',
                'paragraphs': []
            }
        ],
        'version': 'v2.0'
    }

    for context, q_a in context_to_question_answers.items():
        squad['data'][0]['paragraphs'].append({
            'qas': [
                {
                    'question': question,
                    'answers': answers,
                    'is_impossible': False,

                    'id': str(uuid.uuid4()).replace('-', '')
                } for question, answers in q_a.items()
            ],
            'context': context
        })

    return squad, unnatural, incorrect

def get_current_data() -> Tuple[str, str, str]:
    """
    Gets the current data, according to the index_input in the state.

    :return: The context, question, and answer.

    :raises ValueError: If the session state does not contain data and index_input
    """

    if not ('data' in st.session_state and 'index_input' in st.session_state):
        raise ValueError('Session state is not sufficiently initialised. Both "data" and "index_input" should be initialised.')

    return st.session_state['data'][st.session_state['index_input']]

def load_user_profile_and_dataset(username: str, preliminary: bool, subset_index: Optional[int] = None) -> None:
    """
    Read a user's existing annotations, and the full dataset, into memory and stores them in the streamlit session state.

    Also initialises the unsuitable contexts, unsuitable questions, etc.

    :param username: The user's username.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.
    """

    st.session_state['index_input'] = 0
    st.session_state['example_index'] = 0
    st.session_state['data'] = []
    st.session_state['completed_questions'] = set()
    st.session_state['unsuitable_questions'] = collections.defaultdict(list)
    st.session_state['notes'] = {}
    st.session_state['times'] = {
        'examples': {},
        'questions': {}
    }
    st.session_state['errors'] = {
        'question': [],
        'answer': []
    }

    logger.info('Loading user data')

    data_filepath = get_data_filepath(preliminary, subset_index)

    logger.info(f'Looking for data at {data_filepath}')
    if os.path.isfile(data_filepath):
        with open(data_filepath, 'r') as f:
            raw_data = json.load(f)

        for context, qa_data in raw_data.items():
            for question, answer in qa_data.items():
                st.session_state['data'].append( (context, question, answer ) )
    else:
        logger.warning(f'The file "{data_filepath}" does not exist, so we have not loaded any data..')

    st.session_state['data'] = shuffle_qa_data(username, st.session_state['data'])

    with open(get_profile_filepath(username, preliminary, subset_index), 'r', encoding='utf-8') as f:
        st.session_state['kept_pairs'] = json.load(f)

        st.session_state['completed_questions'] = set(row['Question'] for row in st.session_state['kept_pairs'])

    unsuitable_questions_filepath = get_unsuitable_questions_filepath(username, preliminary, subset_index)

    if os.path.isfile(unsuitable_questions_filepath):
        with open(unsuitable_questions_filepath, 'r', encoding='utf-8') as f:
            st.session_state['unsuitable_questions'].update(json.load(f))

    times_filepath = get_times_filepath(username, preliminary, subset_index)

    if os.path.isfile(times_filepath):
        with open(times_filepath, 'r', encoding='utf-8') as f:
            st.session_state['times'].update(json.load(f))

    notes_filepath = get_notes_filepath(username, preliminary, subset_index)

    if os.path.isfile(notes_filepath):
        with open(notes_filepath, 'r', encoding='utf-8') as f:
            st.session_state['notes'].update(json.load(f))

def init_user(username: str, preliminary: bool, subset_index: Optional[int] = None) -> None:
    """
    Initialize an empty user profile for a given username.
        Does nothing if the user already exists.

    :param username: The username for the user.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.
    """

    profile_filepath = get_profile_filepath(username, preliminary, subset_index)

    if os.path.isfile(profile_filepath):
        logger.warning(f'Profile for {username} already exists at {profile_filepath}, not remaking.')
        return

    os.makedirs(os.path.dirname(profile_filepath), exist_ok=True)

    if os.path.isdir(os.path.dirname(profile_filepath)):
        logger.info(f'Made directory for {username}')
    else:
        logger.error(f'Could not make directory for {username}')

    logger.info(f'Making a profile at {profile_filepath} for {username}')

    with open(profile_filepath, 'w') as f:
        json.dump([], f)

def list_users(preliminary: bool, subset_index: Optional[int] = None) -> List[str]:
    """
    Lists the users for which we have profiles.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.

    :return The names of the profiles.
    """

    profiles = []

    # Treat it as a profile filepath in order to filter out extraneous dirs
    glob_expression = get_profile_filepath('*', preliminary, subset_index)

    for filepath in glob.glob(glob_expression):
        dir_path = os.path.dirname(filepath)

        profile_name = dir_path.split(os.path.sep)[-1]

        profiles.append(profile_name)

    return profiles

def render_user_info() -> None:
    """
    If the user is logged in, then show their username and an option to log out, then log them out if they press the button.

    Logging out clears the current user, completed questions, and kept pairs from the state and then reruns the app.
    """

    st.write(f'You are currently logged in as Prolific User "{st.session_state["user"]}"') # TODO Change this to be appropriate to your own annotation management platform.

    logout_is_pressed = st.button('Click here to logout')

    if logout_is_pressed:
        st.session_state['user'] = None
        st.session_state['completed_questions'] = None
        st.session_state['unsuitable_questions'] = collections.defaultdict(list)
        st.session_state['kept_pairs'] = []
        st.session_state['example_index'] = 0
        st.session_state['examples_finished'] = False

        st.session_state['times'] = {
            'examples': {},
            'questions': {}
        }

        st.session_state['errors'] = {
            'question': [],
            'answer': []
        }

        st.session_state['first_user_input'] = True

        st.experimental_rerun()

def render_password_view() -> None:
    """
    Requests a password from the user.

    :raises FileNotFoundError: If the password file cannot be found.
    """

    password_filepath = 'password'

    if os.path.isfile(password_filepath):
        with open('password', 'r') as f:
            password = f.read().strip()
    else:
        raise FileNotFoundError(f'No password file found at "{password_filepath}"')

    st.write('Please input the password to be granted access to the annotation tool.')

    password_input = st.text_input(label='Password', key='password_input', type='password', autocomplete='')

    password_input_hash = hashlib.sha512(password_input.encode()).hexdigest()

    if password_input_hash == password:
        st.session_state['password_given'] = True
        st.experimental_rerun()
    elif len(password_input) > 0:
        st.error('That password is incorrect.')

def run_qa_tool(preliminary: bool, subset_index: Optional[int] = None) -> None:
    """
    This function contains the main annotation tool behaviours

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.
    """

    if preliminary:
        completion_code = 'FILL ME IN' # TODO You should fill this in with your own completion code, if relevant.
    else:
        completion_code = 'FILL ME IN' # TODO You should fill this in with your own completion code, if relevant.

    with st.sidebar:
        render_user_info()
        st.markdown('<hr/>', unsafe_allow_html=True)

        st.sidebar.markdown('**Need a reminder of the judgement guidelines? Check below.**')

        st.sidebar.markdown('<hr style="margin: 0px">', unsafe_allow_html=True)

        with st.sidebar.expander('1. Question Instructions and Examples', expanded=False):
            st.markdown(get_question_instructions_examples(), unsafe_allow_html=True)

        st.markdown('<hr style="margin: 0px">', unsafe_allow_html=True)

        with st.expander('2. Answer Instructions and Examples', expanded=False):
            st.markdown(get_answer_instructions_examples(), unsafe_allow_html=True)

    logger.info(f'User info rendered - {st.session_state["user"]}')

    if ('completed_questions' not in st.session_state) or (st.session_state['completed_questions'] is None):
        load_user_profile_and_dataset(st.session_state['user'], preliminary, subset_index)
        logger.info('Loaded user profile and the dataset')

    new_pair = False
    while st.session_state['index_input'] < len(st.session_state['data']):
        current_context, current_question, current_answer = get_current_data()

        qa_pair_annotated = ( current_question in st.session_state['completed_questions'] )
        qa_pair_unsuitable = ( current_context in st.session_state['unsuitable_questions'] and current_question in st.session_state['unsuitable_questions'][current_context] )

        if qa_pair_annotated or qa_pair_unsuitable:
            st.session_state['index_input'] += 1

            new_pair = True
            
        else:
            break

    if new_pair or 'start_time' not in st.session_state:
        if all(len(error_list) == 0 for error_list in st.session_state['errors'].values()):
            scroll_to_top()

        with st.spinner('Loading...'):
            time.sleep(1)  # Otherwise, it's not always clear that the data's changed, especially when annotating multiple QA pairs from the same context

        st.session_state['start_time'] = time.time()

    if st.session_state['index_input'] == len(st.session_state['data']): # TODO Change this message to be appropriate to the annotation management platform you're using.
        st.success(f'''You have verified all of the data, thank you!
Please enter completion code {completion_code} on Prolific in order to officially complete the study.

Please message us via Prolific if you have any problems with the code.''')

        open(get_completeness_marker_filepath(st.session_state['user'], preliminary, subset_index), 'w').close() # Make the empty file as a marker

        return

    question_suitability_key = f'question_suitability_radio_{st.session_state["index_input"]}'
    question_naturalness_key = f'question_naturalness_checkbox_{st.session_state["index_input"]}'
    question_explanation_key = f'question_explanation_input_{st.session_state["index_input"]}'
    user_question_key = f'question_input_{st.session_state["index_input"]}'

    answer_keys = {
        'naturalness': f'answer_naturalness_checkbox_{st.session_state["index_input"]}',
        'adequacy': f'answer_adequacy_checkbox_{st.session_state["index_input"]}',
        'precision': f'answer_correctness_checkbox_{st.session_state["index_input"]}'
    }
    answer_explanation_key = f'answer_explanation_input_{st.session_state["index_input"]}'
    user_answer_key = f'answer_input_{st.session_state["index_input"]}'

    with st.expander(label='Purpose of this Tool', expanded=False):
        st.write('''This tool allows you to judge whether or not a given question and answer are correct and read naturally, based on a short document.\n
This data can then be used to refine the Artificial Intelligence model which generated them.\n
This model can be used to automatically answer questions posed by users based on documents such as news articles, once fully trained.\n
The more data you judge, the faster and better the model can learn how to do so. 

All of the data shown is entirely computer-generated. 

**The sidebar on the left provides instructions and examples to help if you're unsure how to judge something.**

**Thank you for participating!**''')

    st.markdown(f'## Question-Answer Pair {st.session_state["index_input"] + 1} / {len(st.session_state["data"])}')

    st.markdown('#### Document')

    st.markdown(f'*{current_context}*')

    st.markdown('<hr/>', unsafe_allow_html=True)

    st.markdown('#### Question')

    st.markdown(f'*{current_question}*')

    question_unsuitable_option = 'The original question cannot be answered or is irrelevant'
    question_suitable_option = 'The original question is answerable and relevant'

    question_suitability_options = [question_unsuitable_option, question_suitable_option]

    st.radio(
        label='',
        options=[question_unsuitable_option, question_suitable_option],
        index=question_suitability_options.index(question_suitable_option),
        key=question_suitability_key
    )

    user_question = st.session_state[user_question_key] if user_question_key in st.session_state else current_question

    if st.session_state[question_suitability_key] == question_suitable_option:
        st.checkbox(
            label='The original question reads naturally',
            value=False,
            key=question_naturalness_key
        )

        st.text_area(
            label='Please modify the below question to read naturally, if it doesn\'t already.',
            value=user_question,
            key=user_question_key,
            disabled=st.session_state[question_naturalness_key]
        )
    else:
        st.markdown(f'**You have marked the question as unsuitable, and thus judgements about the question\'s naturalness are not relevant.**')

    st.text_input(
        label='Explanation of your judgement (optional)',
        value='',
        key=question_explanation_key,
        help='If you\'d like to explain the way you judged this question, please feel free, especially if the correct judgement wasn\'t obvious.',
        placeholder='The question cannot be answered because ...'
    )

    for error in st.session_state['errors']['question']:
        st.error(error)

    st.markdown('<hr style="margin: 0px">', unsafe_allow_html=True)

    ########
    # Now handle the answer
    ########

    if st.session_state[question_suitability_key] == question_unsuitable_option:
        st.markdown(f'**You have marked the question as unsuitable, and thus judgements about the answer are not relevant.**')
    else:
        st.markdown('#### Answer')

        st.markdown(f'*{current_answer}*')

        st.markdown('**If you have modified the question, please judge the answer based on the *modified* question.**')

        answer_column_left, answer_column_centre, answer_column_right = st.columns(3)

        with answer_column_left:
            st.checkbox(
                label='The original answer reads naturally',
                value=False,
                key=answer_keys['naturalness']
            )

        with answer_column_centre:
            st.checkbox(
                label='The original answer is adequate',
                value=False,
                key=answer_keys['adequacy']
            )

        with answer_column_right:
            st.checkbox(
                label='The original answer is precise and correct',
                value=False,
                key=answer_keys['precision']
            )

        user_answer = st.session_state[user_answer_key] if user_answer_key in st.session_state else current_answer

        st.text_area(
            label='''Please modify the below answer to read naturally and be more precise/correct, if need be. 
    The answer must be a case-sensitive snippet from the document.''',
            value=user_answer,
            key=user_answer_key,
            disabled=all(st.session_state[answer_key] for answer_key in answer_keys.values()),
        )

        st.text_input(
            label='Explanation of your judgement (optional)',
            value='',
            key=answer_explanation_key,
            help='If you\'d like to explain the way you judged this answer, please do so, especially if the correct judgement wasn\'t obvious.',
            placeholder='The answer is adequate, but imprecise because ...'
        )

        for error in st.session_state['errors']['answer']:
            st.error(error)

    def export_data() -> None:
        squad_pairs, unnatural_questions, incorrect_qa = kept_pairs_to_output(st.session_state['kept_pairs'])

        output_data = [
            (get_profile_filepath, st.session_state['kept_pairs']),
            (get_squad_filepath, squad_pairs),
            (get_unnatural_questions_filepath, unnatural_questions),
            (get_unsuitable_questions_filepath, st.session_state['unsuitable_questions']),
            (get_incorrect_qa_filepath, incorrect_qa),
            (get_times_filepath, st.session_state['times']),
            (get_notes_filepath, st.session_state['notes'])
        ]

        for filepath_func, data_to_export in output_data:
            filepath = filepath_func(st.session_state['user'], preliminary, subset_index)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_export, f, indent=4)

        logger.debug(f'Data saved')

    def add_notes(context: str, question: str, answer: str) -> None:
        question_note = st.session_state[question_explanation_key].strip()

        if answer_explanation_key in st.session_state:
            answer_note = st.session_state[answer_explanation_key].strip()
        else:
            answer_note = ''

        if any(len(note) for note in [question_note, answer_note]):
            if context not in st.session_state['notes']:
                st.session_state['notes'][context] = {}

            st.session_state['notes'][context][question] = {
                'answer': {
                    answer: answer_note if len(answer_note) else None
                },
                'note': question_note if len(question_note) else None
            }

    def save_qa_pair(
        original_question_naturalness: bool,
        original_answer_naturalness: bool,
        original_answer_adequacy: bool,
        original_answer_correctness: bool,
        user_question: str,
        user_answer: str
    ) -> None:
        """
        Saves all the data into kept_pairs in the session state.

        Updates the index_input appropriately.

        :param original_question_naturalness: The naturalness of the original question. True if natural.

        :param original_answer_naturalness: The naturalness of the original answer. True if natural.

        :param original_answer_adequacy: The adequacy of the original answer. True if it is adequate.

        :param original_answer_correctness: The correctness of the original answer. True if it was correct.

        :param user_question: The user's input question. May be identical to the original one.

        :param user_answer: The user's input answer. May be identical to the original one.
        """

        st.session_state['kept_pairs'].append({
            'Context': current_context,
            'Question': current_question,
            'Answer': current_answer,
            'Original Question Naturalness': original_question_naturalness,
            'Original Answer Naturalness': original_answer_naturalness,
            'Original Answer Adequacy': original_answer_adequacy,
            'Original Answer Correctness': original_answer_correctness,
            'User Query': user_question,
            'User Answer': user_answer
        })

        st.session_state['completed_questions'].add(current_question)

    def submit_qa(context: str, question: str, answer: str) -> None:

        st.session_state['errors']['question'] = []
        st.session_state['errors']['answer'] = []

        st.session_state['times']['questions'][st.session_state['index_input']] = time.time() - st.session_state['start_time']

        add_notes(context, question, answer)

        if st.session_state[question_suitability_key] == question_unsuitable_option:
            st.session_state['unsuitable_questions'][context].append(question)

            export_data()
        else:

            submitted_question = st.session_state[user_question_key].strip()
            submitted_answer = st.session_state[user_answer_key].strip()

            question_modified = (submitted_question != current_question.strip())
            question_natural = st.session_state[question_naturalness_key]

            answer_modified = (submitted_answer != current_answer.strip())
            answer_natural = st.session_state[answer_keys['naturalness']]
            answer_adequate = st.session_state[answer_keys['adequacy']]
            answer_correct = st.session_state[answer_keys['precision']]

            if len(submitted_question) == 0:
                st.session_state['errors']['question'].append('The question cannot be blank.')

            if len(submitted_answer) == 0:
                st.session_state['errors']['answer'].append('The answer cannot be blank,')

            if question_modified and question_natural:
                st.session_state['errors']['question'].append(
                    'The question is marked as reading naturally, but has also been modified. Only ones that do not read naturally should be modified.')
            elif not question_modified and not question_natural:
                st.session_state['errors']['question'].append(
                    'The question is marked as not reading naturally, but it has not been modified. Please modify it to read naturally.')

            if submitted_answer not in current_context:
                st.session_state['errors']['answer'].append(f'''The answer "{submitted_answer}" does not appear in the document, please provide an answer that does (case-sensitive).
            If you cannot do so whilst maintaining naturalness and correctness, please mark the question as unsuitable/impossible.''')

            if answer_correct and not answer_adequate:
                st.session_state['errors']['answer'].append(
                    f'Any precise-and-correct answer should also be adequate, but it was not marked as such.')

            if answer_modified:
                if answer_correct and answer_natural:
                    st.session_state['errors']['answer'].append(
                        'The answer is marked as precise, correct and reading naturally, but has been modified. Only ones with problems should be modified.')
            else:
                if not answer_natural:
                    st.session_state['errors']['answer'].append(
                        'The answer is marked as not reading naturally, but has not been modified. Please modify it to read naturally.')

                if not answer_correct and not answer_adequate:
                    st.session_state['errors']['answer'].append(
                        'The answer is marked as incorrect, but has not been modified. Please modify it to be correct.')

            if all(len(error_list) == 0 for error_list in st.session_state['errors'].values()):
                save_qa_pair(
                    question_natural,
                    answer_natural,
                    answer_adequate,
                    answer_correct,
                    st.session_state[user_question_key],
                    st.session_state[user_answer_key]
                )

                export_data()

    st.button(label='Submit judgements', on_click=lambda: submit_qa(current_context, current_question, current_answer))

def run_calibration(preliminary: bool, subset_index: Optional[int] = None) -> None:
    """
    Runs "user calibration" by giving the users instructions and examples on how to judge things.

    :param preliminary: Whether or not we are running in preliminary mode.

    :param subset_index: Optional. If given, indicates that we are running on this subset of data.
    """
    scroll_to_top()

    if ('completed_questions' not in st.session_state) or (st.session_state['completed_questions'] is None):
        load_user_profile_and_dataset(st.session_state['user'], preliminary, subset_index)
        logger.info('Loaded user profile and the dataset')

    examples = [
        ('#### 1. Judging Questions', get_question_instructions_examples()),
        ('#### 2. Judging Answers', get_answer_instructions_examples()),
    ]

    start_time = time.time()

    def update_time():
        if st.session_state['example_index'] not in st.session_state['times']['examples']:
            st.session_state['times']['examples'][st.session_state['example_index']] = 0

        st.session_state['times']['examples'][st.session_state['example_index']] += time.time() - start_time

        filepath = get_times_filepath(st.session_state['user'], preliminary, subset_index)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(st.session_state['times'], f, indent=4)

    def next_example():
        update_time()

        st.session_state['example_index'] += 1

        if st.session_state['example_index'] == len(examples):
            st.session_state['examples_finished'] = True
            scroll_to_top()

    def previous_example():
        update_time()

        st.session_state['example_index'] -= 1

    with st.sidebar:
        render_user_info()

    st.markdown(f'## Instructions ({st.session_state["example_index"] + 1} / {len(examples)})')

    st.markdown('This tool will ask you to judge the quality, naturalness, and correctness of a series of Question-Answer pairs each associated with a short document. This step is designed to demonstrate the kind of judgements we\'re looking for and guide you through the process.')

    st.markdown('<hr/>', unsafe_allow_html=True)

    example_title, example_text = examples[st.session_state['example_index']]

    st.markdown(example_title)
    st.markdown(example_text, unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)

    is_last_example = (st.session_state['example_index'] == len(examples) - 1)

    if is_last_example:
        st.markdown('**If you need to see this guidance again during the annotation process, you can find it in the sidebar on the left.**')

    left_column, right_column = st.columns(2)

    with left_column:
        if st.session_state['example_index'] > 0:
            st.button(f'Previous ({st.session_state["example_index"]} / {len(examples)})', on_click=previous_example)

    with right_column:
        button_text = 'Start Judgements' if is_last_example else f'Next ({st.session_state["example_index"] + 2} / {len(examples)})'

        st.button(button_text, on_click=next_example)

def show_login_view(preliminary: bool, subset_index: Optional[int] = None) -> None:
    """
    Shows a view where users can select a username (or create a new user).

    Once a user is selected or created, the app is restarted with that user loaded.

    :param preliminary: Whether or not we're running in preliminary mode.

    :param subset_index: Optional. The subset index to use. If given, indicates that we're running on a subset of the data.
    """

    blank_option = '---'
    create_user_option = 'Create User'

    profiles = list_users(preliminary, subset_index)

    completed_users = [profile for profile in profiles if
                       os.path.isfile(get_completeness_marker_filepath(profile, preliminary, subset_index))]

    # Otherwise, you can game the system by logging in as a user who's completed it and getting the completion code.
    incomplete_users = [profile for profile in profiles if profile not in completed_users]

    options = [blank_option, *incomplete_users, create_user_option]

    if len(profiles):
        options.insert(-1, blank_option)

    if 'first_user_input' not in st.session_state:
        st.session_state['first_user_input'] = True

    username = st.text_input(
        label='Please enter your Prolific ID', # TODO Change this to be appropriate to your own annotation management platform.
    )

    if len(username.strip()) == 0:
        if 'first_user_input' in st.session_state and not st.session_state['first_user_input']:
            st.error('Empty IDs are not valid.')
    elif username in completed_users:
        # TODO Change this to be appropriate to your own annotation management platform.
        st.error(f'You have already completed the study and cannot login again. Please contact us on Prolific if you\'re experiencing issues.')
    else:
        init_user(username, preliminary, subset_index)
        st.session_state['user'] = username

        st.experimental_rerun()

    st.session_state['first_user_input'] = False

    with st.expander(label='Purpose of this Tool', expanded=True):
        st.write('''This tool allows you to judge whether or not a given question and answer read naturally and are correct, based on a short document.\n
This data can then be used to refine the Artificial Intelligence model which generated them.\n
This model can be used to automatically answer questions posed by users based on documents such as news articles, once fully trained.\n
The more data you judge, the faster and better the model can learn how to do so. 

All of the data shown is entirely computer-generated. 

**Thank you for participating!**''')

def main() -> None:
    """
    Runs the QA tool.

    First asks for a password. Then a user selection. Then finally if both password and user are provided, run the tool proper.
    """

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument('--preliminary', action='store_true', help='If set, we use the preliminary dataset rather than the full dataset.')
    group.add_argument('--subset', type=int, choices=list(range(1, 101)), nargs=1, help='The index of the dataset subset to use. If omitted, the full dataset is used.')

    parser.add_argument('--insecure', action='store_true', help=f'If set, we run in insecure mode, which means you don\'t need a password file.')

    args = parser.parse_args()
    preliminary = args.preliminary

    subset = args.subset[0] if args.subset is not None else args.subset

    insecure = args.insecure

    if 'first_run' not in st.session_state or st.session_state['first_run']:
        if preliminary:
            logger.info('Running in preliminary mode')

        if insecure:
            logger.warning(f'Running in insecure mode. Not asking for a password.')

        if 'password_given' not in st.session_state:
            st.session_state['password_given'] = False

    if not insecure and not st.session_state['password_given']:
        render_password_view()
    elif ('user' not in st.session_state) or (st.session_state['user'] is None):
        show_login_view(preliminary, subset)
    elif 'examples_finished' not in st.session_state or not st.session_state['examples_finished']:
        run_calibration(preliminary, subset)
    elif not os.path.isfile(get_profile_filepath(st.session_state['user'], preliminary, subset)):
        # TODO Change this to be appropriate to your own annotation management platform.
        st.error('We appear to be experiencing technical issues. Please refresh and try again (any progress will be saved). If they persist, please contact us on Prolific.')
    else:
        run_qa_tool(preliminary, subset)

    if 'first_run' not in st.session_state:
        st.session_state['first_run'] = False

if __name__ == '__main__':
    main()
