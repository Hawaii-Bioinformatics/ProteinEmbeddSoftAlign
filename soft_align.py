# Load libraries
from collections import defaultdict
from Bio import SeqIO
import pandas as pd
import torch.nn  as nn
import torch
import networkx as nx
from io import StringIO
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
top_n = 3
min_diagonal_length = 10
max_mismatches = 3

# The input of this function are two dictionaries (seq_1 and seq_2) containing the sequence representations
i = 0
def get_data_matrix(seq_1, seq_2):
    x_tensor = seq_1["representations"][36]
    y_tensor = seq_2["representations"][36]

    # Normalize the vectors (this is needed for cosine similarity)
    x_norm = x_tensor / x_tensor.norm(dim=1)[:, None]
    y_norm = y_tensor / y_tensor.norm(dim=1)[:, None]

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(x_norm, y_norm.transpose(0,1))

    # If you need the output as a DataFrame
    data = pd.DataFrame(cosine_similarity_matrix.numpy())
    return data


def find_mutual_matches(data):
    """
    find_mutual_matches takes 'data' dataframe containing cosine
    distance matrix of seq_1, seq_2.

    If a cosine distance is a top_n distance in a row
    and is also a top_n distance in the column that cell
    is considered mutual matches.

    Mutual matches are then stored in 'matches' set.
    """
    # Find the top_n distances in each row.
    rows = pd.DataFrame({n: data.T[col].nlargest(top_n).index.tolist()
                         for n, col in enumerate(data.T)}).T

    # Find the top_n distances in each column.
    cols = pd.DataFrame({n: data[row].nlargest(top_n).index.tolist()
                         for n, row in enumerate(data)})

    matches = set()

    # Loop over top_n in the rows and columns to find mutual matches.
    for i, n_cols in enumerate(rows.values):
        for c in n_cols:
            if i in cols.iloc[:, c].values:
                matches.add((i, int(c)))

    return matches


def add_matching_neighbors(seq_1_str, seq_2_str, matches):
    """
    seq_1_str, seq_2_str are strings containing the sequence to the
    corresponding seq_1,seq_2 sequence IDs.

    matches is the set that is passed from the find_mutual_matches
    function.

    add_matching_neighbors function considers neighbors to matches
    with identical amino acids as matches and adds them to the
    'matches' set.
    """
    temp_set = set()

    for match in matches:
        if match[0] > 0 and match[1] > 0 and (seq_1_str[match[0] - 1] == seq_2_str[match[1] - 1]):
            temp_set.add((match[0] - 1, match[1] - 1))

        if match[0] < len(seq_1_str) - 1 and match[1] < len(seq_2_str) - 1 and (
                seq_1_str[match[0] + 1] == seq_2_str[match[1] + 1]):
            temp_set.add((match[0] + 1, match[1] + 1))

    matches = matches.union(temp_set)

    return matches


def find_exclusive_intervals(intervals):
    """
    intervals = found_matches
    found_matches stores matches of identical amino acids when
    sequence strings are rotated.

    find_exclusive_intervals returns found_matches that were not
    identified in by other match finding functions.
    """
    exclusive_intervals = []

    for i in intervals:
        is_included = False

        for j in intervals:
            if i != j and i[0] >= j[0] and i[1] <= j[1]:
                is_included = True
                break

        if not is_included:
            exclusive_intervals.append(i)

    return exclusive_intervals


def find_matches(s, t, offset_val, matches, k, nb_errors=2):
    """
    s = seq_1_str
    t = seq_2_str
    offset_value = l or r rotation offset value
    matches = 'matches' set
    k = max_mismatches

    find_matches searches through rotated sequence alignments and if
    amino acids in the same positions are identical, they are considered
    a match.
    """

    found_matches = []

    max_errors_available = nb_errors

    i = 0
    end = 1
    nb_matches = 0
    while i < len(s) - k:
        start = i
        # check if the current character in s matches the current character in t
        if s[i] == t[i] or (i, i + offset_val) in matches:
            nb_matches += 1
            # nested loop to iterate through the rest of the characters in s
            for j in range(i + 1, len(s)):
                # check if the current character in s matches the current character in t
                if s[j] == t[j] or (j, j + offset_val) in matches:
                    end = j
                    nb_matches += 1
                else:
                    # decrement the number of errors allowed in the potential match
                    max_errors_available -= 1

                # check if the number of errors encountered so far is
                # greater than the allowed number
                if max_errors_available < 0:
                    # check if the potential match is at least max_mismatches characters long
                    if nb_matches >= k:
                        # add the match to the found_matches list
                        found_matches.append((start, end))

                    # reset the number matches
                    nb_matches = 0
                    # reset the number of errors allowed in the potential match
                    max_errors_available = nb_errors
                    # update the outer loop index and potential match start and end indices
                    i += 1
                    start = i
                    end = i + 1
                    break
            # check if the potential match is at least max_mismatches characters long
            if nb_matches >= k:
                # add the match to the found_matches list
                found_matches.append((start, end))
            # update the outer loop index and potential match start and end indices
            i += 1
            start = i
            end = i + 1
        else:
            i += 1

        # combine matches included in other matches
        unique_found_matches = find_exclusive_intervals(found_matches)

    return unique_found_matches

def get_matches(seq_1_str, seq_2_str, data, max_mismatches=3):
    matches = find_mutual_matches(data)
    matches = add_matching_neighbors(seq_1_str, seq_2_str, matches)
    valid_segments = find_all_matches(seq_1_str, seq_2_str, max_mismatches, matches)
    valid_segments = sorted(valid_segments, key=lambda x: x[0][0])
    valid_diagonals = get_valid_diagonals(valid_segments)
    matches = cleanup_matches(matches, valid_diagonals)
    return matches


def generate_rrotation(s, t, offset):
    """
    generate_lrotation inputs:
    s = seq_1_str
    t = seq_2_str
    offset = position in sequence where offset occurs

    generate_lrotation function rotates seq_2_str 1 position right
    along corresponding seq_1_str for each iteration and
    returns rotated string.
    """
    # If the offset is larger than the length of the
    # sequence 't', raise an exception.
    if offset >= len(s):
        raise Exception(f"offset {offset} larger than seq length {len(s)}")

    lgaps = '-' * offset

    # Extract a substring from sequence 't' starting from the offset
    # index up to the length of 's'.
    # my_str represents the part of 't' that will be kept after the rotation.
    my_str = t[0:len(s) - offset]

    # Generate a string of '-' characters of length equal to the remaining
    # length of 's' after adding 'my_str'.
    # rgaps represents the right gaps that will be added to the end of the sequence.
    rgaps = '-' * (len(s) - len(lgaps + my_str))

    return lgaps + my_str + rgaps


def generate_lrotation(s, t, offset):
    """
    generate_lrotation inputs:
    s = seq_1_str
    t = seq_2_str
    offset = position in sequence where offset occurs

    generate_lrotation function rotates seq_2_str 1 position left
    along corresponding seq_1_str for each iteration and
    returns rotated string.
    """
    # If the offset is larger than the length of the
    # sequence 't', raise an exception.
    if offset >= len(t):
        raise Exception(f"offset {offset} larger than seq length {len(s)}")

    # Extract a substring from sequence 't' starting from the offset
    # index up to the length of 's'.
    # my_str represents the part of 't' that will be kept after the rotation.
    my_str = t[offset:len(s)]

    # Generate a string of '-' characters of length equal to the remaining
    # length of 's' after adding 'my_str'.
    # rgaps represents the right gaps that will be added to the end of the sequence.
    rgaps = '-' * (len(s) - len(my_str))

    return my_str + rgaps


def find_all_matches(s, t, k, matched_pairs):
    """
    find_all_matches inputs:
    s = seq_1 sequence string denoted as 'seq_1_str'
    t = seq_2 sequence string denoted as 'seq_2_str'
    k = max_mismatches, hyperparameter defined above for amount of
    mismatches allowed.
    matched_pairs = current 'matches' list, which contains mutual matches
    and matching neighbors.
    """
    all_matches = []

    # In each iteration, generate a right rotation of 'seq_2_str' by the
    # current index and run find_match function to identify matching pairs
    # in 'seq_1_str' and 'seq_2_str' after rotation.
    # Matched pairs identified during rotation are added to all_matches
    # list.
    for i in range(0, len(s)):
        t_offset = generate_rrotation(s, t, i)

        match_in_i = find_matches(s, t_offset, -i, matched_pairs, k)

        # Adds another match along the same diagonal to match_in_i
        match_in_j = [(x - i, y - i) for x, y in match_in_i]

        # Adds both matches along same diagonal to 'all_matches' list
        all_matches.extend(list(zip(match_in_i, match_in_j)))

    # In each iteration, generate a left rotation of 'seq_2_str' by the
    # current index and run find_match function to identify matching pairs
    # in 'seq_1_str' and 'seq_2_str' after rotation.
    # Matched pairs identified during rotation are added to all_matches
    # list.
    for i in range(1, len(t)):
        t_offset = generate_lrotation(s, t, i)

        match_in_i = find_matches(s, t_offset, +i, matched_pairs, k)

        # Adds another match along the same diagonal to match_in_i
        match_in_j = [(x + i, y + i) for x, y in match_in_i]

        # Adds both matches along same diagonal to 'all_matches' list
        all_matches.extend(list(zip(match_in_i, match_in_j)))

    return all_matches


def build_paths_graph(data, matches):
    """
    build_paths_graph function identifies diagonal segments
    from sorted matches.
    """
    dag = {}

    graph = nx.DiGraph()

    max_depth = max([x[0] for x in matches])

    # Sort the matches based on the second element of the match pairs.
    sorted_matches = sorted(matches, key=lambda x: x[1])

    # Loop over the sorted matches and
    # add edges between them to build the graph.
    for i in range(len(sorted_matches) - 1):
        last_depth = max_depth
        dag[sorted_matches[i]] = []

        for j in range(i + 1, len(sorted_matches)):

            if (sorted_matches[i][0] == sorted_matches[j][0]) or (sorted_matches[i][1] == sorted_matches[j][1]):
                # Don't consider overlapping cells
                continue

            if (sorted_matches[j][0]) < last_depth and (sorted_matches[j][0] > sorted_matches[i][0]):
                dag[sorted_matches[i]].append(sorted_matches[j])
                seq_1_idx, seq_2_idx = sorted_matches[j]
                graph.add_edge(sorted_matches[i], sorted_matches[j], weigth=data.iloc[seq_1_idx, seq_2_idx])
                last_depth = sorted_matches[j][0]

    return graph


def get_valid_diagonals(valid_segments):
    """
    valid_segments = sorted(valid_segments)

    get_valid_diagonals function identifies matches that occur consecutively
    in a diagonal and stores them in a dictionary 'valid_diagonals'.
    """
    valid_diagonals = defaultdict(int)

    # Loop over the valid segments and add the length of each segment
    # to its corresponding diagonal in the dictionary.
    for x in valid_segments:
        min_val = min(x[0][0], x[1][0])
        diag = (x[0][0] - min_val, x[1][0] - min_val)
        valid_diagonals[diag] += x[0][1] - x[0][0] + 1

    return valid_diagonals


def cleanup_matches(matches, valid_diagonals):
    """
    cleanup_matches removes matches that do not occur in a valid_diagonal
    but are shorter than min_diagonal_length (hyperparameter).
    """
    remove_elems = []

    # Loop over the matches and add any invalid match to the removal list
    for x in matches:
        min_val = min(x[0], x[1])
        diag = (x[0] - min_val, x[1] - min_val)
        if valid_diagonals[diag] < min_diagonal_length:
            remove_elems.append(x)

    # Remove the invalid matches from the original list
    matches = list(set(matches).difference(remove_elems))

    return matches


def get_longest_path(data, matches):
    longest_path = []

    # If there are any matches left, build a paths graph and find the longest path in the graph
    if len(matches) > 0:
        graph = build_paths_graph(data, matches)
        longest_path = nx.dag_longest_path(graph)

    return longest_path

def soft_align(seq_1_str, seq_2_str, seq_1_embedding, seq_2_embedding):
    data = get_data_matrix(seq_1_embedding, seq_2_embedding)
    matches = get_matches(seq_1_str, seq_2_str, data)
    longest_path = get_longest_path(data, matches)
    return longest_path


def find_homologous_pos(seq_1_str, seq_1_pos, seq_2_str, seq_1_embedding, seq_2_embedding):
    data = get_data_matrix(seq_1_embedding, seq_2_embedding)
    matches = get_matches(seq_1_str, seq_2_str, data)
    longest_path = get_longest_path(data, matches)
    longest_path_dict = dict(longest_path)
    return longest_path_dict.get(seq_1_pos, None)
