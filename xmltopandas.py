# Flatten XML to pandas creating a column for each element with nested element tags included in the column name

from collections import defaultdict
import xml.etree.ElementTree as ET
from os import listdir
import pandas as pd


# flatten xml tree into one row with as many columns as needed
def flatten_xml(node, key_prefix=()):

    # grab element tag
    text = (node.text or '').strip()
    if text:
        yield key_prefix, text

    # copy attributes
    for attr, value in node.items():
        yield key_prefix + (attr,), value

    # recurse into children
    for child in node:
        yield from flatten_xml(child, key_prefix + (child.tag,))

# convert key pairs into a dictionary
def key_pairs_to_dict(pairs, key_sep='-'):

    out = {}

    # group by key 
    key_map = defaultdict(list)
    for key_parts, value in pairs:
        key_map[key_sep.join(key_parts)].append(value)

    # create dict 
    for key, values in key_map.items():
        if len(values) == 1:  # No need to suffix keys.
            out[key] = values[0]
        else:  # More than one value for this key.
            for suffix, value in enumerate(values, 1):
                out[f'{key}{key_sep}{suffix}'] = value

    return out
    
main_df = pd.DataFrame()

# change below to the dir where your data lives
# grab each xml file in the dir (one level), parse each into a row and append to a pandas df
for file in listdir("data_path"):
    with open("data_path/"+file, 'r', encoding="utf-8") as content:
        tree = ET.parse(content)
        root = tree.getroot()
        # flatten xml into rows
        rows = [key_pairs_to_dict(flatten_xml(row)) for row in root]
        # convert to pandas
        df = pd.DataFrame(rows)
        main_df = pd.concat([main_df, df], ignore_index=True)
        
main_df
