import pandas as pd
import numpy as np
import re
import copy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pprint

###
# Simple method for getting dataset
# Made this to avoid retyping code
###
def get_dataset(name, drop_duplicates=False, **args):

    """
        Retrieves the dataset specified by name.
        name should be one of:
        "case_data", "person_data", "case_person_data", "interventions"
        drop_duplicates: if True, drop global duplicates before returning dataframe
        args: any other args to pass to pd.read_csv
    """

    strs = {
        "case_data": "../data/processed/case-data.csv",
        "person_data": "../data/processed/person-data.csv",
        "case_person_data": "../data/processed/case-person-data.csv",
        "interventions": "../data/raw/extracted/interventions.csv",
    }

    df = pd.read_csv(strs[name], **args)
    if drop_duplicates:
        df = df.drop_duplicates()
    return df

###################
## DATA CLEANING ##
###################

# Duplicate Labels identified by domain knowledge
# The labels in the lists will be mapped down to their keys
duplicates_1 = {
    "Direct Contact": ["Client contact in office", "Direct Contact through Outreach", "Client contact", "Client Contact out of office"],
    "Attempted client contact": ["Attempted client contact through Outreach", "Attempt to locate client"],
    "Client Assessment Conducted": ["New Client assessment completed"],
    "Coordinated Assessment Completed at DACC": ["Coordinated Assessment Completed through HOST contact"]
}

# Domain Knowledge + additional duplicates based on data exploration.
duplicates_2 = {
    "Direct Contact": ["Client contact in office", "Direct Contact through Outreach", "Client contact", "Client Contact out of office"],
    "Attempted client contact": ["Attempted client contact through Outreach", "Attempt to locate client"],
    "Client Assessment Conducted": ["New Client assessment completed"],
    "Coordinated Assessment Completed at DACC": ["Coordinated Assessment Completed through HOST contact", "Coordinated Assessment Scheduled at DACC"],
    "Release of information forms signed for all appropriate vendors and agencies": ["Client declined to sign release of information"]
}

# Words to which to map similar terms.
# The strings in the lists will be mapped down to their keys.
# need to think about whether the keys should be abberviations or full words
# to provide model with full context. 
word_replace_dict = {
    "birth certificate": ["birth cert", "birth certificate", "bc"],
    "texas id": ["texas id", "tx id", "state id", "texas state id", "tx state id"],
    "_CM_": ["case manager", "casemanager", "writer", "cm"],
    "case manager": ["_CM_", "this _CM_"],
    "coordinated assessment": ["coordinated assessment"],
    "social security card": ["ss", "ssc"],
    "ROI": ["release of information", "rois"],
    "client": ["cl", "client", "clt", "peer"],
}

# Labels describing client contact
contact_labels = ["Direct Contact", "Collateral Contact", "Attempted client contact", "No Show"]

# Based on data exploration, labels identified to have little to no correlation between note text and label
bad_labels = ["Client not assigned DACC CSR due to court order",
              "Asked client to complete a DACC Customer survey",
              "HMIS Release of Information - Declined"]

def replace_words(text, word_replace_dict):
    """
    Helper method for replacing words in a given string based on a word replacement dictionary.
    Check format for word_replace_dict in default dict provided above.
    """
    for replacement, words in word_replace_dict.items():
        # \b is word boundary
        # https://stackoverflow.com/questions/54481198/what-is-a-better-way-to-match-multiple-substrings-in-a-string-than-to-search-for
        pat = r'\b({})\b'.format('|'.join(words))
        text = re.sub(pat, replacement, text, flags=re.IGNORECASE)
    return text

# Word_replace_dict values will be case ignorant.
# Turned this into a general purpose preprocessing method
def get_clf_data(drop_global_duplicates=False, remove_irrelvant_cols=True, drop_empty_notes=True, 
                 drop_interventions=True, duplicate_labels_dict=None, word_replace_dict=None, 
                 labels_to_ignore=None):

    """
    General purpose retrival and cleaning method for checkbox data. Uses case-data.csv for the initial dataframe.
    This can be easily repurposed to be a general purpose cleaning method for any multi-label dataset.

    drop_global_duplicates: If true, drop global duplicates in dataframe before any data cleaning. 
    remove_irrelvant_cols: If true, Only keep NOTEID, NOTE, DESCRIPTION, and GROUPNAME cols
    drop_empty_notes: If true, drop any entries whose notes are empty strings
    drop_interventions: Drop interventions data by only keeping the entries whose GROUPNAME cols are "Team Check-In-->" or "ID Documents"
    duplicate_labels_dict: If provided, consolidate duplicate labels based on provided dict.
    word_replace_dict: If provided, for each text column, map words based on provided dict.
    labels_to_ignore: After duplicate labels are dropped, drops any entries with the provided labels. 
    """

    print("--- LOADING DATA... ---")

    df = get_dataset("case_data", drop_duplicates=drop_global_duplicates)
    df = df.astype({"NOTE": str})

    print("Total initial entries: %d" % len(df.index))
    print("Total initial notes: %d" % df.groupby("NOTEID").ngroups)

    if remove_irrelvant_cols:
        df = df[["NOTEID", "NOTE", "DESCRIPTION", "GROUPNAME"]]
        df.drop_duplicates(inplace=True)
        print("- Removed irrelevant columns for checkbox classification, kept:", df.columns)

    # Empty notes are of no use
    if drop_empty_notes:
        df = df[df["NOTE"].str.strip().astype(bool)]
        df['NOTE'] = df['NOTE'].replace('', np.nan)
        df = df.dropna(subset=["NOTE"])
        print("- Dropped empty notes")

    # Removing interventions checkboxes
    if drop_interventions:
        df = df[df["GROUPNAME"].str.contains("Team Check-In-->|ID Documents-->")]
        df = df.drop("GROUPNAME", axis=1)
        df["DESCRIPTION"] = df["DESCRIPTION"].astype("category")
        df = df.sort_values(by="NOTEID")
        print("- Dropped interventions checkboxes")

    # Combine the following. According to interviews with CMs, while they would 
    # be fine with a dropdown for different types of Client Contact (in office, outreach, etc.)
    # They all ultimately just refer to the same thing. 

    # https://stackoverflow.com/questions/32262982/pandas-combining-multiple-categories-into-one

    if duplicate_labels_dict:
        mappings = pd.Series(duplicate_labels_dict).explode().sort_values()
        mappings_dict = dict(zip(mappings.values, mappings.index))
        df["DESCRIPTION"] = df["DESCRIPTION"].apply(lambda x: mappings_dict.get(x, x).strip())

        # After remapping, Make sure same checkbox is not checked twice for a note
        df.drop_duplicates(["NOTEID", "DESCRIPTION"], inplace=True)
        print("- Consolidated duplicate labels using the provided mappings.")
        print(duplicate_labels_dict)

    if labels_to_ignore:
        df = df.loc[~df["DESCRIPTION"].isin(labels_to_ignore)]
        print("- Dropped entries containing the provided labels")
    
    if word_replace_dict:
        new_notes = df.set_index("NOTEID")["NOTE"].apply(lambda x: replace_words(x, word_replace_dict))
        df.set_index("NOTEID", inplace=True)
        df["NOTE"] = new_notes
        df.reset_index(inplace=True)
        print("- Substituted occurences of words in text with provided word mappings")
        print(word_replace_dict)

    print("Total labels: %d" % len(df["DESCRIPTION"].unique()))
    print("Total entries after preprocessing: %d" % len(df.index))
    print("Total notes after preprocessing: %d" % df.groupby("NOTEID").ngroups)

    print("--- FINISHED LOADING DATA. ---")

    return df.reset_index(drop=True)

##############
# UPSAMPLING #
##############

# from cb temp mixing
def temperature_scaled_mixing(df: pd.DataFrame, label_col: str, T, K=None, frac=1.0, replace=True):
    """
    df: the dataset to sample
    label_col: the column containing the labels
    T: The temperature parameter. When T=1, this is identical to examples-proportional mixing
    K: The artificial size limit. If not provided, defaults to size of largest label set
    frac: What fraction of the original df should the returned sampled df be. If 1, len(sampled.index) == len(df.index)
    replace: Whether to replace samples after sampling. 
    """

    label_counts = df[label_col].value_counts()
    if not K:
        K = label_counts.max() 
    total_labels = label_counts.apply(lambda x: min(x, K)).sum()
    weights = label_counts.apply(lambda x: min(x, K) / total_labels)

    weights = weights.pow(1.0 / T)
    weights /= weights.sum()
    
    df.set_index(label_col, inplace=True)
    df["class_weights"] = weights
    df["class_totals"] = label_counts
    df["sample_weights"] = df["class_weights"] / df["class_totals"]
    df.reset_index(inplace=True)
    
    sampled = df.sample(frac=frac, weights="sample_weights", replace=replace)
    sampled.reset_index(drop=True)
    
    df.drop(columns=["class_weights", "class_totals", "sample_weights"], inplace=True)
    sampled.drop(columns=["class_weights", "class_totals", "sample_weights"], inplace=True)

    return sampled    

#### A heuristic I'm using for upsampling ####

def min_descriptions(df: pd.DataFrame, label_col, min_label_col, duplicate_id_col):
    """
    Adds a column that contains the label from the labelset for the note that occurs the fewest times across the dataset
    """

    # For each row, add a column that contains the number of times the checkbox for that row was checked.
    # Ex. Note: ..., Description: No Show, FREQ: 1893 
    df["FREQ"] = df.groupby(label_col)[label_col].transform('count')

    # For each note, the description column now contains the label from the note's labelset
    # that is checked the least amount of times across the dataset.
    min_label = df.sort_values("FREQ").drop_duplicates(duplicate_id_col).set_index(duplicate_id_col)[label_col]

    # Merge back to entire dataset
    df = df.set_index(duplicate_id_col).merge(min_label.rename(min_label_col), left_index=True, right_index=True)

    return df.drop("FREQ", axis=1).reset_index(drop=False)

######################
## ONE HOT ENCODING ##
######################

def one_hot_encode(df: pd.DataFrame, label_col, new_col, duplicate_id_col, drop_old_label_col=False, classes=None):
    """
    For a multi-label dataset where each note has duplicate entries that differ by label_col, returns
    a onehot encoded dataset using sklearn.MultiLabelBinarizer. 

    Make sure label_col and new_col are different
    """

    mlb = MultiLabelBinarizer()
    already_fitted = False
    if classes is None:
        already_fitted = True
        classes = df[label_col].sort_values().unique()
        mlb.fit([classes])

    # A list of lists, where each inner list contains all labels for a note
    rows = df.set_index(duplicate_id_col).groupby(duplicate_id_col)[label_col].apply(list)
    sorted_ids = rows.index
    sorted_labels = rows.values    

    binarized = mlb.transform(sorted_labels) if already_fitted else mlb.fit_transform(sorted_labels)
    onehot = pd.Series(dict(zip(sorted_ids, binarized)))

    onehot_df = df.set_index(duplicate_id_col, drop=False).merge(onehot.rename(new_col), left_index=True, right_index=True)
    onehot_df = onehot_df.drop_duplicates(duplicate_id_col).reset_index(drop=True)

    if drop_old_label_col:
        onehot_df = onehot_df.drop(label_col, axis=1)

    return (onehot_df, classes)

#### Count onehot labels ####

def onehot_label_counts(df, onehot_col, classes):
    """
    Akin to using .value_counts() on the label column before onehot encoding
    """
    labels_lists = np.array(df[onehot_col].tolist())
    counts = [0] * len(classes)
    for ll in labels_lists:
        counts += ll
    return dict(zip(classes, counts))

#### REVERSE ONE HOT ####

def onehot_to_label_names(df, label_list_col, label_names):
    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    label_names_lists = mlb.inverse_transform(np.stack(df[label_list_col]))
    return label_names_lists

# Set index to id col before using (like noteid)
def explode_onehot_df(df: pd.DataFrame, label_list_col, label_names,
                      exploded_col_name, keep_label_names=False):
    """
    Undoes onehot explosion
    """

    if not isinstance(df[label_list_col].iloc[0], np.ndarray):
        print("ERROR: label_list_col must contain onehot encoded lists...")
        return df
    
    df.sort_index(inplace=True)
    sorted_ids = df.index
    label_names_lists = onehot_to_label_names(df, label_list_col, label_names)
    label_names_series = pd.Series(dict(zip(sorted_ids, label_names_lists))).rename(exploded_col_name)

    df = df.merge(label_names_series, left_index=True, right_index=True)

    # For seeing all other labels checked on a note during qualitative analysis
    if not keep_label_names:
        df = df.drop(label_list_col, axis=1)

    df = df.explode(exploded_col_name)
    return df

##################
### ENTAILMENT ###
##################

ID = "id"
ENTAILMENT = "entailment"
NOT_ENTAILMENT = "not_entailment"
CONTRADICTION = "contradiction"
NEUTRAL = "neutral"
PREMISE = "premise"
HYPOTHESIS = "hypothesis"

GOLD_2 = "gold_2"
GOLD_3 = "gold_3"
GOLD_2_IDX = GOLD_2 + "_idx"
GOLD_3_IDX = GOLD_3 + "_idx"

two_label_set = [ENTAILMENT, NOT_ENTAILMENT]
two_label_dict = {val: i for i, val in enumerate(two_label_set)}

three_label_set = [ENTAILMENT, NEUTRAL, CONTRADICTION]
three_label_dict = {val: i for i, val in enumerate(three_label_set)}

def set_contradiction(row, contra_set):
    label = row[HYPOTHESIS]
    return CONTRADICTION if label in contra_set else row[GOLD_3]

def make_entailment_set(df: pd.DataFrame, premise_col, hypothesis_col, id_col, 
                        contra_dict=None, aug_dict=None, add_two_label_set=False):
    """
    Creates an entailment dataset out of a multilabel text-label pair dataset where each 
    label for a text is represented with a duplicated entry with a different value in the 
    label column. 

    premise_col: column containing the text. Gets renamed to "premise" in output
    hypothesis_col: column containing the label. Gets renamed to "hypothesis" in output
    id_col: the column which contains an id that's duplicated for each text entry. Gets renamed to "id" in output
    contra_dict: a dictionary containing contradiction information from domain knowledge about the dataset
    aug_dict: a dictionary containing mappings from labels to augmented label text. 
    add_two_label_set: By default, this returns a three label entailment set. Set this to true to add a column with a two_label entailment labels.

    output df structure:
    columns: id, premise, hypothesis, gold_3, gold_3_idx, (optional: gold_2, gold_2_idx), ... other columns in original df
    """

    # Restructure
    df = df.rename(columns={id_col: ID, premise_col: PREMISE, hypothesis_col: HYPOTHESIS})
    df = df.reset_index(drop=True)

    # Step 1: Basic set
    df[GOLD_3] = ENTAILMENT
    df_cols = df.columns
    new_cols = [HYPOTHESIS, GOLD_3] # Cols which will have new values for the new entries 
    old_cols = [col for col in df.columns if col not in new_cols] # Cols which will have duplicated vals for the new entries
    basic_set = df

    # Step 2: Add every casenote-label pair possible with "contradiction" or "neutral" as appropriate
    labels_list = df[HYPOTHESIS].unique()
    grouped = basic_set.groupby(ID)
    df_list = [basic_set]
    for name, group in grouped:
        existing_labels = group[HYPOTHESIS].tolist()
        remaining_labels = [label for label in labels_list 
                            if label not in existing_labels]

        new_df = pd.DataFrame(columns=df_cols)

        new_df[HYPOTHESIS] = pd.Series(remaining_labels)
        new_df[GOLD_3] = NEUTRAL

        # Preserve values of old cols in new rows
        # We already know that the values of these old cols will be the same for each note
        for col in old_cols:
            new_df[col] = group[col].iloc[0]

        if contra_dict:
            contradictions_set = set()
            for ex_lab in existing_labels:
                contradictions_set.update(contra_dict.get(ex_lab, [ex_lab]))

        new_df[GOLD_3] = new_df.apply(lambda x: set_contradiction(x, contradictions_set), axis=1)

        df_list.append(new_df)

    full_set = pd.concat(df_list)

    # I fixed the error in my code causing duplicates, but leaving this here just in case.
    full_set = full_set.drop_duplicates()

    # Add label indicies 
    full_set[GOLD_3_IDX] = full_set[GOLD_3].apply(lambda x: three_label_dict[x])

    if aug_dict:
        full_set[HYPOTHESIS] = full_set[HYPOTHESIS].apply(lambda x: aug_dict.get(x, x)) 

    if add_two_label_set:
        full_set[GOLD_2] = full_set[GOLD_3].apply(lambda x: NOT_ENTAILMENT if x != ENTAILMENT else ENTAILMENT)
        full_set[GOLD_2_IDX] = full_set[GOLD_2].apply(lambda x: two_label_dict[x])


    return full_set 

### CLASSIFICATION REPORT ###

PREDS = "preds"
ONEHOT_GOLDS = "onehot_golds"
ONEHOT_PREDS = "onehot_preds"

# Testing method for turning a dataset with entailment predictions into a classification report
def entailment_clf_report(test_df: pd.DataFrame, model_outputs, classes, aug_rev_dict=None):
    """
    For an entailment-classification set made using make_entailment_set,
    Given the test set and the models final outputs, print the classification report.

    test_df: df containing test set. Test set should contain
    num_notes * num_labels entries, such that when you group by the id_col,
    there are num_labels entries for each note. Make sure id_col (or any df column)
    is not the current index. 
    
    model_outputs: An array of len(test_df.index) such that the i-th row in model_outputs
    corresponds to the i-th row in test_df. Make sure to NOT shuffle the testing data
    loader to ensure this is the case. 

    id_col: id column that's duplicated for each note-label pair.
    aug_rev_dict: if provided, will map the hypothesis according to the mappings in the dict.
    """

    test_df = test_df.reset_index(drop=True) # reset index to 0 -> len(test_df.index)
    test_df[PREDS] = pd.Series(model_outputs) # so this works as intended

    if aug_rev_dict:
        test_df[HYPOTHESIS] = test_df[HYPOTHESIS].apply(lambda x: aug_rev_dict.get(x, x))

    # entailment index is always 0
    # Either 2 or 3 labelset works.
    
    # The actual labels checked (anything that's not entailment was not part of note labels)
    golds_df = test_df[test_df[GOLD_3_IDX] == 0].reset_index(drop=True)

    # The predictions (anything that model predicts as entailment is a label prediction)
    preds_df = test_df[test_df[PREDS] == 0].reset_index(drop=True)

    # It's possible that preds has fewer notes (unique ids) because some notes never had an entailment prediction
    
    # Need to provide classes because its possible that certain classes were never predicted as entailment by model
    onehot_df = one_hot_encode(golds_df, HYPOTHESIS, ONEHOT_GOLDS, ID, classes=classes)[0]
    empty_preds = pd.Series(dict(zip(onehot_df[ID].tolist(), np.zeros(shape=(len(onehot_df.index), len(classes))))))
    onehot_df = onehot_df.set_index(ID).merge(empty_preds.rename(ONEHOT_PREDS), left_index=True, right_index=True)
    
    onehot_preds_df = one_hot_encode(preds_df, HYPOTHESIS, ONEHOT_PREDS, ID, classes=classes)[0].set_index(ID)
    onehot_df[ONEHOT_PREDS].update(onehot_preds_df[ONEHOT_PREDS])

    y_true = onehot_df[ONEHOT_GOLDS].tolist()
    y_pred = onehot_df[ONEHOT_PREDS].tolist()

    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

### Split Entailment set into test and train ###

FULL = "full"
CONTRA = "contra"
BASIC = "basic"

def get_basic_set(df):
    # For the provided entailment df, only keep the 
    # entailment entries. (identical to simply adding a 
    # column containing "entailment" for each entry in original non-entailment dataset)
    return df.loc[df[GOLD_3] == ENTAILMENT]

def get_contradictions_set(df):
    # For the provided entailment df, only keep the
    # entailment and contradction entries (not neutrals)

    # If using the two_label_set for training, this teaches the model
    # that only contradictions are not_entailment, and there's not
    # data about the neutrals.
    return df.loc[df[GOLD_3] != NEUTRAL]

# This was originally intended just for entailment sets, can actually split and mix
# both classification and entailment sets the way its written. 
def split_mix_set(df: pd.DataFrame, frac=1, test_size=0.2, random_state=42, mix_train_params=None):
    """
    Regardless of train set, the test set needs to contain every possible note-label pair
    for each note in the test set. 

    df: entailment set obtained with make_entailment_set
    mix_train_params: If not none, temperature mix training set with provided (T, K, label_col) params.
    T: temperature, K: artifical size limit for labels of one class, label_col: the col containing the labels to upsample using. 

    test_size: fraction of unique notes in test set from full set. 
    frac: what fraction of unique notes to retain in train and test set (after splitting and mixing, if applicable)
    random_state: iykyk
    """

    # percentage of casenotes to sample
    # num_samples = int(len(unique_ids) * frac)
    # sampled_noteids = np.random.RandomState(seed=random_state).choice(unique_ids, num_samples)
    
    unique_ids_df = df.drop_duplicates(ID)
    train_notes, test_notes = train_test_split(unique_ids_df, test_size=test_size, random_state=random_state)

    if mix_train_params:
        T, K, label_col = mix_train_params
        train_notes = temperature_scaled_mixing(train_notes, label_col, T, K, frac)
    else: 
        train_notes = train_notes.sample(frac=frac, random_state=random_state)

    train_notes = train_notes[ID].tolist()
    test_notes = test_notes.sample(frac=frac, random_state=random_state)[ID].tolist()

    # Train notes can have duplicate ids
    grouped = df.groupby(ID)
    train_dfs = []
    for name, group in grouped:
        count = train_notes.count(name)
        for _ in range(count):
            train_dfs.append(group)
    train_set = pd.concat(train_dfs)

    test_set = df.loc[df[ID].isin(test_notes)]

    print("Total notes: %d" % len(unique_ids_df.index))
    print("Num Notes in Train Set: %d, Test Set: %d" % (len(train_notes), len(test_notes)))
    print("Total Entries: %d" % (len(train_set.index) + len(test_set.index)))
    print("Num Entries in Train: %d, Test: %d" % (len(train_set.index), len(test_set.index)))

    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)

##################
### DEPRACATED ###
##################

# From cb_baseline_clf.ipynb
# ensure that duplicate_id_col is not the index when you pass the df

### my_one_hot_encode works, but using MultiLabelBinarizer version in favor of this.
def my_multilabel_binarizer(labels_list, classes):
    class_map = {label:i for i, label in enumerate(classes)}

    binarized = []
    dup_lists = 0
    for labels in labels_list:
        onehot_list = [0] * len(classes)
        if len(labels) != len(set(labels)):
            dup_lists += 1

        for label in labels:
            # If I change to += 1, the value_counts are same as before onehot_encoding 
            # If I set it equal to 1, the value_counts of Direct, Collateral, and Attempted contacts are lower
            onehot_list[class_map[label]] = 1
        binarized.append(onehot_list)
    
    return binarized

def my_one_hot_encode(df: pd.DataFrame, label_col, new_col, duplicate_id_col, classes=None):

    if not classes:
        classes = df[label_col].sort_values().unique()

    # A list of lists, where each inner list contains all labels for a note
    rows = df.set_index(duplicate_id_col).groupby(duplicate_id_col)[label_col].apply(list)
    sorted_ids = rows.index
    sorted_labels = rows.values    

    binarized = my_multilabel_binarizer(sorted_labels, classes)
    onehot = pd.Series(dict(zip(sorted_ids, binarized)))

    onehot_df = df.set_index(duplicate_id_col, drop=False).merge(onehot.rename(new_col), left_index=True, right_index=True)
    onehot_df = onehot_df.drop_duplicates(duplicate_id_col).reset_index(drop=True)
    return (onehot_df, classes)

# Returns a tuple containing the data frame and label names in their corresponding locations
def _one_hot_encode_labels_as_list(df: pd.DataFrame, label_col, new_labels_list_col, dup_id_col):
    """
    df: the dataframe containing the labels to one hot encode
    label_col: the column containing the categorical vars to one hot encode
    new_labels_list_col: the col where the new lists for each unique row will be stored
    dup_id_col: the id col that's the same for duplicate rows where only the value in the label_col is different
    """

    one_hot_labels = pd.get_dummies(df[label_col])
    label_names = one_hot_labels.columns.tolist()
    oh_lists = one_hot_labels.apply(lambda x: list(x), axis=1)
    # At this point, no duplicates have been dropped. Instead, the duplicated rows' description values
    # have been one hot encoded. 

    ohl_df = df.drop(label_col, axis=1)
    ohl_df[new_labels_list_col] = oh_lists

    # NOTEID is the same for rows with same note content
    # https://stackoverflow.com/questions/51926668/pandas-groupby-aggregate-element-wise-list-addition
    grouped = ohl_df.groupby(dup_id_col)[new_labels_list_col].apply(lambda x: [sum(y) for y in zip(*x)])
    ohl_df.drop_duplicates(subset=dup_id_col, inplace=True)
    ohl_df.set_index(dup_id_col, drop=False, inplace=True)
    ohl_df[new_labels_list_col] = grouped

    ohl_df.reset_index(inplace=True, drop=True)

    return (ohl_df, label_names)

# Deprecated
def _one_hot_encode_labels_as_columns(df: pd.DataFrame, label_col, dup_id_col):
    """
    df: the dataframe containing the labels to one hot encode
    label_col: the column containing the categorical vars to one hot encode
    dup_id_col: the id col that's the same for duplicate rows where only the value in the label_col is different
    """
    one_hot_labels = pd.get_dummies(df["DESCRIPTION"])
    label_names = one_hot_labels.columns.tolist()

    oh_df = df.drop(label_col, axis=1)
    oh_df = oh_df.join(one_hot_labels)
    oh_df = oh_df.groupby(dup_id_col).sum().reset_index()
    
    return (oh_df, label_names)