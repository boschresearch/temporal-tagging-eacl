""" Utility classes and functions related to the multilingual temporal tagging (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH
@author: Lukas Lange

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import os
import re


class Rule:
    def __init__(self, name, extraction, value, quant="", freq="", mod=""):
        self.name = name
        self.extraction = extraction
        self.value = value
        self.quant = quant
        self.freq = freq
        self.mod = mod

    def __str__(self):
        s = 'RULENAME="' + self.name + '"'
        s += ',EXTRACTION="' + self.extraction + '"'
        s += ',NORM_VALUE="' + self.value + '"'
        if self.quant != "":
            s += ',NORM_QUANT="' + self.quant + '"'
        if self.freq != "":
            s += ',NORM_FREQ="' + self.freq + '"'
        if self.mod != "":
            s += ',NORM_MOD="' + self.mod + '"'
        return s

    def __repr__(self):
        return str(self)


RULE_NAME_REGEX = 'RULENAME="(.*?)"(,|$)'
RULE_EXTRACTION_REGEX = 'EXTRACTION="(.*?)"(,|$)'
RULE_NORM_VALUE_REGEX = 'NORM_VALUE="(.*?)"(,|$)'
RULE_NORM_QUANT_REGEX = 'NORM_QUANT="(.*?)"(,|$)'
RULE_NORM_FREQ_REGEX = 'NORM_FREQ="(.*?)"(,|$)'
RULE_NORM_MOD_REGEX = 'NORM_MOD="(.*?)"(,|$)'


def _read_pattern_from_file(path, encoding="utf-8"):
    disjunctions = []
    try:
        with open(path, "r", encoding=encoding) as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                else:
                    disjunctions.append(line.replace("(", "(?:"))
        # assert len(disjunctions) > 0, path
        name = (
            path.split("/")[-1].replace("resources_repattern_", "").replace(".txt", "")
        )
        pattern = "(" + "|".join(disjunctions) + ")"
        return name, pattern
    except UnicodeDecodeError:
        return _read_pattern_from_file(path, "latin1")


def _read_rules_from_file(path, encoding="utf-8"):
    rules = {}
    try:
        with open(path, "r", encoding=encoding) as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                else:
                    m_name = re.search(RULE_NAME_REGEX, line)
                    m_extraction = re.search(RULE_EXTRACTION_REGEX, line)
                    m_value = re.search(RULE_NORM_VALUE_REGEX, line)
                    m_quant = re.search(RULE_NORM_QUANT_REGEX, line)
                    m_freq = re.search(RULE_NORM_FREQ_REGEX, line)
                    m_mod = re.search(RULE_NORM_MOD_REGEX, line)

                    assert m_name and m_extraction and m_value
                    name = m_name.group(1)
                    extraction = m_extraction.group(1)
                    value = m_value.group(1)
                    rule = Rule(name, extraction, value)

                    if m_quant:
                        rule.quant = m_quant.group(1)
                    if m_freq:
                        rule.freq = m_freq.group(1)
                    if m_mod:
                        rule.mod = m_mod.group(1)
                    rules[name] = rule
        return rules
    except UnicodeDecodeError:
        return _read_rules_from_file(path, "latin1")


def _read_normalization_from_file(path, encoding="utf-8"):
    normalization = {}
    try:
        with open(path, "r", encoding=encoding) as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                else:
                    try:
                        text, norm = line.split('","')
                        text = text.strip()[1:]
                        norm = norm.strip()[:-1]
                        normalization[text] = norm
                    except:
                        continue

        return normalization
    except UnicodeDecodeError:
        return _read_normalization_from_file(path, "latin1")


def _read_elements_from_directory(directory):
    elements = {}
    for fname in os.listdir(directory):
        path = directory + fname

        if "/repattern" in directory:
            read_function = _read_pattern_from_file
            name, content = read_function(path)
            elements[name] = content

        elif "/rules" in directory:
            if "interval" in path:
                continue
            read_function = _read_rules_from_file
            for e, v in read_function(path).items():
                elements[e] = v

        elif "/normalization" in path:
            read_function = _read_normalization_from_file
            content = read_function(path)
            name = fname.replace("resources_normalization_norm", "re").replace(
                ".txt", ""
            )
            elements[name] = content

    return elements


def expand_regex(item_dict):
    terms_to_check = [(x, x, y) for x, y in item_dict.items()]
    out = {}

    while len(terms_to_check) > 0:
        x = terms_to_check.pop(0)
        term_regex, term_org, norm_value = x

        m_sq = re.search("\[(.*?)\](\?)?", term_regex)
        m_rd = re.search("\((.*?)\)(\?)?", term_regex)

        if m_sq:
            term_begin = term_regex[: m_sq.start()]
            term_end = term_regex[m_sq.end() :]
            items = m_sq.group(1)
            optional = m_sq.group(2) == "?"

            for char in items:
                new_term = term_begin + char + term_end
                y = new_term, term_org, norm_value
                terms_to_check.append(y)

            if optional:
                new_term = term_begin + term_end
                y = new_term, term_org, norm_value
                terms_to_check.append(y)

        elif m_rd:
            term_begin = term_regex[: m_rd.start()]
            term_end = term_regex[m_rd.end() :]
            items = m_rd.group(1)
            optional = m_rd.group(2) == "?"

            for item in items.split("|"):
                new_term = term_begin + item + term_end
                y = new_term, term_org, norm_value
                terms_to_check.append(y)

            if optional:
                new_term = term_begin + term_end
                y = new_term, term_org, norm_value
                terms_to_check.append(y)

        else:
            if term_regex in out:
                if out[term_regex] != norm_value:
                    print("Norm Value Conflict for: " + term_org)
                    print(out[term_regex], norm_value)
            else:
                out[term_regex] = norm_value

    return out


def split_rules(rules, patterns, convert_numbers=True):
    for r_name, rule in rules.items():
        if convert_numbers:
            rule.extraction = rule.extraction.replace("(\d\d[\d]?[\d]?)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d\d\d\d)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d\d\d0)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d\d\d)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d\d0)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d\d)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d0)", "%reNumber")
            rule.extraction = rule.extraction.replace("(\d)", "%reNumber")
            rule.extraction = rule.extraction.replace("([\d]+)", "%reNumber")
            rule.extraction = rule.extraction.replace("([0-9])", "%reNumber")
            rule.extraction = rule.extraction.replace("(-[\d]+)", "%reNegNumber")
        rule_parts = {}
        rule_regex = rule.extraction
        new_rule = rule_regex

        last_offset = 0
        cur_index = 1

        while "%re" in new_rule:
            offset_bra = rule_regex[last_offset:].find("(")
            offset_pct = rule_regex[last_offset:].find("%re")

            offset_bra = (
                len(rule_regex) + 1 if offset_bra < 0 else last_offset + offset_bra
            )
            offset_pct = (
                len(rule_regex) + 1 if offset_pct < 0 else last_offset + offset_pct
            )

            if offset_pct < offset_bra:
                # next comes a resource pattern
                found = False
                for p_name in sorted(
                    patterns, key=lambda p_name: len(p_name), reverse=True
                ):
                    pattern = patterns[p_name]

                    if (
                        p_name
                        == rule_regex[offset_pct + 1 : offset_pct + 1 + len(p_name)]
                    ):
                        new_rule = new_rule.replace("%" + p_name, pattern, 1)
                        rule_parts[cur_index] = p_name
                        last_offset = offset_pct + 1
                        cur_index += 1
                        found = True
                        break
                assert found

            else:
                m_bra = re.match("\((.*?)\)", rule_regex[offset_bra:])
                assert m_bra

                info = ""
                next_index = cur_index + 1
                components = m_bra.group(1).split("|")
                for c in components:
                    if c.startswith("%re"):
                        info += str(next_index)
                        next_index += 1
                    info += "|"
                rule_parts[cur_index] = info[:-1]
                last_offset = offset_bra + 1
                cur_index += 1
        rule.org_regex = rule_regex
        rule.extraction = new_rule
        rule.parts = rule_parts
        rules[r_name] = rule
    return rules


def get_resources(path_to_language_files: str) -> tuple[any]:
    if path_to_language_files[-1] != "/":
        path_to_language_files += "/"

    print("Load HeidelTime resources from " + path_to_language_files)
    rules = _read_elements_from_directory(path_to_language_files + "rules/")
    patterns = _read_elements_from_directory(path_to_language_files + "repattern/")
    patterns["reNumber"] = "(\d+)"
    patterns["reNegNumber"] = "(-\d+)"
    rules = split_rules(rules, patterns)

    normalization = _read_elements_from_directory(
        path_to_language_files + "normalization/"
    )

    if "reMonthInSeason" not in normalization:
        normalization["reMonthInSeason"] = {}
    normalization["reMonthInSeason"]["01"] = "WI"
    normalization["reMonthInSeason"]["02"] = "WI"
    normalization["reMonthInSeason"]["03"] = "SP"
    normalization["reMonthInSeason"]["04"] = "SP"
    normalization["reMonthInSeason"]["05"] = "SP"
    normalization["reMonthInSeason"]["06"] = "SU"
    normalization["reMonthInSeason"]["07"] = "SU"
    normalization["reMonthInSeason"]["08"] = "SU"
    normalization["reMonthInSeason"]["09"] = "FA"
    normalization["reMonthInSeason"]["10"] = "FA"
    normalization["reMonthInSeason"]["11"] = "FA"
    normalization["reMonthInSeason"]["12"] = "WI"

    if "reMonthInQuarter" not in normalization:
        normalization["reMonthInQuarter"] = {}
    normalization["reMonthInQuarter"]["01"] = "1"
    normalization["reMonthInQuarter"]["02"] = "1"
    normalization["reMonthInQuarter"]["03"] = "1"
    normalization["reMonthInQuarter"]["04"] = "2"
    normalization["reMonthInQuarter"]["05"] = "2"
    normalization["reMonthInQuarter"]["06"] = "2"
    normalization["reMonthInQuarter"]["07"] = "3"
    normalization["reMonthInQuarter"]["08"] = "3"
    normalization["reMonthInQuarter"]["09"] = "3"
    normalization["reMonthInQuarter"]["10"] = "4"
    normalization["reMonthInQuarter"]["11"] = "4"
    normalization["reMonthInQuarter"]["12"] = "4"

    # Add basic information
    if "reWeekday" not in normalization:
        normalization["reWeekday"] = {}
    normalization["reWeekday"]["2"] = "monday"
    normalization["reWeekday"]["3"] = "tuesday"
    normalization["reWeekday"]["4"] = "wednesday"
    normalization["reWeekday"]["5"] = "thursday"
    normalization["reWeekday"]["6"] = "friday"
    normalization["reWeekday"]["7"] = "saturday"
    normalization["reWeekday"]["1"] = "sunday"

    normalization["reWeekdayToInt"] = {}
    normalization["reWeekdayToInt"]["monday"] = 2
    normalization["reWeekdayToInt"]["tuesday"] = 3
    normalization["reWeekdayToInt"]["wednesday"] = 4
    normalization["reWeekdayToInt"]["thursday"] = 5
    normalization["reWeekdayToInt"]["friday"] = 6
    normalization["reWeekdayToInt"]["saturday"] = 7
    normalization["reWeekdayToInt"]["sunday"] = 1

    if "reMonth" not in normalization:
        normalization["reMonth"] = {}
    normalization["reMonth"]["january"] = "01"
    normalization["reMonth"]["february"] = "02"
    normalization["reMonth"]["march"] = "03"
    normalization["reMonth"]["april"] = "04"
    normalization["reMonth"]["may"] = "05"
    normalization["reMonth"]["june"] = "06"
    normalization["reMonth"]["july"] = "07"
    normalization["reMonth"]["august"] = "08"
    normalization["reMonth"]["september"] = "09"
    normalization["reMonth"]["october"] = "10"
    normalization["reMonth"]["november"] = "11"
    normalization["reMonth"]["december"] = "12"

    if "reDurationNumber" in normalization:
        normalization["reNumWord"] = normalization["reDurationNumber"]

    normalization["reYear"] = {}
    for i in range(0, 10000):
        normalization["reYear"][str(i)] = str(i).zfill(4)
        normalization["reYear"][str(i).zfill(4)] = str(i).zfill(4)

    normalization["reYear2Digit"] = {}
    for i in range(0, 100):
        normalization["reYear2Digit"][str(i)] = str(i).zfill(2)
        normalization["reYear2Digit"][str(i).zfill(2)] = str(i).zfill(2)

    normalization["reNumber"] = {}
    for i in range(0, 1000):
        normalization["reNumber"][str(i)] = str(i)

    normalization["reNegNumber"] = {}
    for i in range(0, 1000):
        normalization["reNegNumber"][str(i)] = str(-i)

    normalization["reTimeHour"] = {}
    for i in range(0, 25):
        normalization["reTimeHour"][str(i)] = str(i).zfill(2)
        normalization["reTimeHour"][str(i).zfill(2)] = str(i).zfill(2)

    normalization["reTimeMinute"] = {}
    for i in range(0, 60):
        normalization["reTimeMinute"][str(i)] = str(i).zfill(2)
        normalization["reTimeMinute"][str(i).zfill(2)] = str(i).zfill(2)

    normalization["reTimeSecond"] = {}
    for i in range(0, 60):
        normalization["reTimeSecond"][str(i)] = str(i).zfill(2)
        normalization["reTimeSecond"][str(i).zfill(2)] = str(i).zfill(2)

    normalization["reTime"] = {}
    for i in range(0, 60):
        normalization["reTime"][str(i)] = str(i).zfill(2)
        normalization["reTime"][str(i).zfill(2)] = str(i).zfill(2)

    for key, norm_dict in normalization.items():
        expanded_dict = expand_regex(norm_dict)
        normalization[key] = expanded_dict

    return patterns, normalization, rules
