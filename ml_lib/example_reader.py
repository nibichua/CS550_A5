'''
Created on Apr 25, 2015

@author: mroch
'''

import csv
import re

# For parsing text of format:  verbosevalue=abbrvalue
assign_re = re.compile("(?P<verbose>[^=]+)=(?P<abbrev>.*)")

def csv_data(attr_names, examples, attr_abbr=None, delimiter=','):
    """csv_data reader
    Reads in comma separated value (CSV) files describing a data set. Filenames:
        attr_names - Single-line CSV file with names of attributes
        examples - Multi-line CSV file where each line is an example.
            First value is a category followed by attribute values
            that are presented in the same order as attr_names
        
        attr_abbr - Some datasets use abbreviations for values.  If 
        there is a desire to be able to translate these to more readable
        names at a later date, a CSV file may be specified.  It should
        contain one line for each attribute in the same order as the attributes
        appeared in attr_names.  A multi-level dictionary will be built that 
        permits mapping abbreviations to full names.  e.g. for the UCI mushroom
        dataset:  data['abbr']['ring-type']['c'] returns "cobwebby"
        
        Returns data dictionary with the following keys:
        attributes, abbrev (if attr_abbr file given), examples,
        categories:
            attributes - names of attributes (possibly abbreviated)
            abbrev - full names of attributes
            examples - List attribute lists for each example
            categories - List of category names for each example
            
    """
    
    data = {}
    # attribute names file should contain one row of comma separated
    # names of attriubtes
    with open(attr_names, 'rb') as fileh:
        reader = csv.reader(fileh, delimiter=delimiter)
        data['attributes'] = reader.next()

    
    # attribute data should contain comma separated attributes; 
    # one line for each example
    with open(examples) as fileh:
        reader = csv.reader(fileh, delimiter=delimiter)
        data['examples'] = []
        data['categories'] = []
        for row in reader:
            # First item is label, rest is attributes
            label = row.pop(0)
            data['categories'].append(label)            
            row.append(label)  # Make last entry category
            data['examples'].append(row)
    
    # Create dictionary that permits us to see the full attribute name
    # for each value associated with an attribute.
    if attr_abbr != None:
        data['abbrev'] = {}  # new dictionary
        with open(attr_abbr, 'rb') as fileh:
            reader = csv.reader(fileh, delimiter=delimiter)
            for (attr, values) in zip(data['attributes'], reader):
                # Create dictionary for this attribute
                data['abbrev'][attr] = {}
                # split value=value-abbreviation
                for v in values:
                    m = assign_re.match(v)
                    if not m:
                        return ValueError("Bad attribute abbreviation: %s"%(v))
                    data['abbrev'][attr][m.group("abbrev")] = m.group("verbose")                    

    return data
