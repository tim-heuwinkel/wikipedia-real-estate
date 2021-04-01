#!/usr/bin/env python3
"""
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite

Modified by: Tim Heuwinkel
"""

import re

CONST_VALID_COORDINATE_CHARS = "0123456789.-"
CONST_VALID_COORDINATE_DIRECTIONS = "NSEW"


def extract_text(article):
    """Removes the text from the article body and changes any xml syntax to its human-readable corollary"""

    start = article.find("<text xml:space=\"preserve\">") + len("<text xml:space=\"preserve\">")
    end = article.find("</text>", start)
    text = article[start:end]

    text = text.replace("&quot;", "\"")
    text = text.replace("&amp;", "&")
    text = text.replace("&apos;", "\'")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&nbsp;", " ")

    return text


def extract_hyperlinks(text):
    """Extracts all hyperlinks from the text body and stores them in a python list"""

    hyperlinks = set()
    index = 0

    # Iterate through whole article
    while index < len(text):

        # If a letter is '['
        if text[index] == "[":
            # If the letter following it is also '[', then it is a hyperlink
            if index + 1 < len(text) and text[index + 1] == "[":

                # Build hyperlink
                index += 2
                link = ""
                while index < len(text) and not text[index] == "]" and not text[index] == "|":
                    link += text[index]
                    index += 1
                hyperlinks.add(link)

        index += 1

    # Sort to make searchable via binary search
    hyperlinks = list(hyperlinks)
    hyperlinks.sort()

    return hyperlinks


def validate_coordinate_string(string):
    """Returns whether the given string is comprised of all characters which are valid for numerical coordinates"""

    for char in string:
        if char not in CONST_VALID_COORDINATE_CHARS:
            return False
    if string == "" or string == "." or string == "-" or ".." in string or "--" in string:
        return False

    return True


def to_number(string):
    """Transforms a numerical string to either an int or a float"""

    if "." in string:
        return float(string)
    return int(string)


def scale_checker(string, index):
    # Check to see if "scale" tag follows
    if len(string[index + 1:]) >= 4:
        if string[index + 1: index + 5].lower() == "name":
            return True

    elif len(string[index + 1:]) >= 5:
        if string[index + 1: index + 6].lower() == "scale":
            return True

    elif len(string[index + 1:]) >= 6:
        if string[index + 1: index + 7].lower() == "source":
            return True

    else:
        return False


def parse_coordinates(string):
    """Parses the string coordinate into a numerical representation; works on one degree-minute-second-direction
    sequence; assumes that the string begins with the degree numbers and not '|' """

    # Parse the primary data point, either degrees or latitude
    prev = 0
    index = string.find("|")
    # Iterate through "|...|" segments to find the start of the coordinate sequence
    while not validate_coordinate_string(string[prev:index]):
        prev = index + 1
        if prev >= len(string):
            return None, None, None, None, None
        index = string.find("|", prev)
        if index == -1:
            return None, None, None, None, None

    data1 = to_number(string[prev:index])
    # If it contains no minute or second data (won't be tripped by lat-long format)
    if string[index + 1].upper() in CONST_VALID_COORDINATE_DIRECTIONS:
        return data1, 0, 0, string[index + 1].upper(), index + 3

    # Parse the secondary data point, either minutes or longitude;
    # the second data point is the one that must be tested for lat-long format
    index2 = string.find("|", index + 1)

    if not validate_coordinate_string(string[index + 1:index2]):
        return None, None, None, None, None
    data2 = to_number(string[index + 1:index2])

    if index2 == len(string)-1:
        # coords with lat-long format and hanging "|"
        return data1, data2, None, None, -1

    # If no tertiary data is contained (must check for "scale", etc. edge case)
    if string[index2 + 1].upper() in CONST_VALID_COORDINATE_DIRECTIONS:
        if scale_checker(string, index2):
            return data1, data2, None, None, -1
        else:
            return data1, data2, 0, string[index2 + 1].upper(), index2 + 3

    # If coordinates contain seconds
    elif string[index2 + 1] in CONST_VALID_COORDINATE_CHARS:
        index3 = string.find("|", index2 + 1)

        if not validate_coordinate_string(string[index2 + 1:index3]):
            return None, None, None, None, None
        data3 = to_number(string[index2 + 1:index3])

        if index3 == len(string) - 1:
            # coords with unusual format and or hanging "|"
            return None, None, None, None, None

        # Check to see if "scale" tag follows
        if scale_checker(string, index3):
            return data1, data2, None, None, -1
        else:
            return data1, data2, data3, string[index3 + 1].upper(), index3 + 3

    # If the coordinates are in lat-long format (i.e. a non-NSEW AND non-numeric tag follows 2 numbers)
    else:
        return data1, data2, None, None, -1


def extract_coordinates(article):
    """Extracts the coordinates related to the article in (LAT, LONG) form. If no coordinates are available
    returns 'None'"""

    # Search for "{{[cC]oord|" in the article
    for word in re.finditer("{{[cC]oord\|", article):

        index = word.end()
        brace_count = 0

        # Track the number of open braces to find the end of the coordinate string
        while brace_count >= 0 and index < len(article):
            if article[index] == "{":
                brace_count += 1
            elif article[index] == "}":
                brace_count -= 1

            index += 1

        if index > len(article):
            continue

        # Element at index - 1 will ALWAYS be a "}"
        coordinate_string = article[word.end():index - 1]
        # Replace any newlines, spaces, etc, that may appear
        coordinate_string = coordinate_string.replace("\n", "")
        coordinate_string = coordinate_string.replace(" ", "")
        coordinate_string = coordinate_string.replace("\t", "")
        coordinate_string = coordinate_string.replace("|||", "|")
        coordinate_string = coordinate_string.replace("||", "|")

        # Search the string for coordinates

        data1, data2, seconds1, direction1, end = parse_coordinates(coordinate_string)
        # If something went wrong in the parsing, go to next tag
        if data1 is None:
            continue
        # If the coordinates were structured as (lat, long)
        if end == -1:
            return data1, data2
        degrees2, minutes2, seconds2, direction2, _ = parse_coordinates(coordinate_string[end:])
        # If something went wrong in the parsing, go to next coordinate tag
        if degrees2 is None or minutes2 is None or seconds2 is None or direction2 is None:
            continue

        # incorporate minutes and seconds into degrees
        lat = data1 + data2 / 60 + seconds1 / 3600
        long = degrees2 + minutes2 / 60 + seconds2 / 3600

        if direction1 == "S":
            lat = lat * -1
        if direction2 == "W":
            long = long * -1

        return lat, long

    # "{{[cC]oord|" not found in article
    return None
