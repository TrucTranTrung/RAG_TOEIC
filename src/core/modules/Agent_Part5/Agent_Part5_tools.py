import requests
from langchain_core.tools import tool
from typing import Dict, List, Optional


@tool
def vocab_search(word: str, full: bool = False) -> dict:
    """
    Searches for vocabulary definitions from an online dictionary API.
    Useful for checking word meanings, parts of speech, and usage examples.
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()[0]

        results = []
        meanings = data.get("meanings", [])
        if not full:
            meanings = meanings[:2]

        for m in meanings:
            defs = m.get("definitions", [])
            if not defs:
                continue

            d = defs[0]  
            results.append({
                "partOfSpeech": m.get("partOfSpeech"),
                "definition": d.get("definition"),
                "example": d.get("example")
            })

        return {
            "word": word,
            "meanings": results
        }

    except Exception:
        return {
            "word": word,
            "meanings": [],
            "error": "Definition not found"
        }


@tool
def grammar_pattern_lookup(pattern_type: str) -> dict:
    """
    Lookup common grammar patterns and structures for TOEIC Part 5.
    
    Args:
        pattern_type: Type of grammar pattern to lookup. Options:
            - "verb_patterns": Common verb + to-inf or verb + V-ing patterns
            - "prepositions": Common preposition usage after verbs/adjectives
            - "word_forms": Rules for noun/verb/adjective/adverb forms
            - "conjunctions": Coordinating and subordinating conjunctions
            - "relative_pronouns": Usage of who/which/that/where/when
            - "conditionals": If-clause patterns
            - "comparatives": Comparative and superlative forms
    
    Returns:
        Dictionary with pattern rules and examples
    """
    
    patterns = {
        "verb_patterns": {
            "to_infinitive": {
                "verbs": ["want", "decide", "plan", "hope", "expect", "need", "would like", "agree", "refuse", "promise"],
                "structure": "verb + to + infinitive",
                "example": "The company plans to expand next year."
            },
            "gerund": {
                "verbs": ["enjoy", "finish", "avoid", "consider", "suggest", "practice", "mind", "keep", "delay", "risk"],
                "structure": "verb + V-ing",
                "example": "We enjoy working with the team."
            },
            "both": {
                "verbs": ["start", "begin", "continue", "like", "love", "hate", "prefer"],
                "note": "These verbs can take both to-inf and V-ing with little difference in meaning"
            }
        },
        
        "prepositions": {
            "depend_on": "The decision depends on the manager's approval.",
            "interested_in": "She is interested in marketing.",
            "responsible_for": "He is responsible for the project.",
            "good_at": "She is good at programming.",
            "famous_for": "The company is famous for innovation.",
            "different_from": "This model is different from the previous one.",
            "worried_about": "We are worried about the deadline.",
            "proud_of": "The team is proud of their achievement."
        },
        
        "word_forms": {
            "noun_suffixes": ["-tion", "-ment", "-ness", "-ity", "-er/-or", "-ance/-ence"],
            "adjective_suffixes": ["-ful", "-less", "-ous", "-ive", "-able/-ible", "-al"],
            "adverb_suffix": ["-ly"],
            "verb_suffixes": ["-ize", "-ify", "-en"],
            "rules": {
                "subject_position": "Usually needs a noun",
                "after_article": "Use noun (a/an/the + noun)",
                "before_noun": "Use adjective (adjective + noun)",
                "modify_verb": "Use adverb (verb + adverb)"
            }
        },
        
        "conjunctions": {
            "coordinating": {
                "FANBOYS": ["for", "and", "nor", "but", "or", "yet", "so"],
                "usage": "Connect equal grammatical elements"
            },
            "subordinating": {
                "time": ["when", "while", "before", "after", "as soon as", "until", "since"],
                "reason": ["because", "since", "as"],
                "contrast": ["although", "though", "even though", "whereas", "while"],
                "condition": ["if", "unless", "provided that", "as long as"],
                "purpose": ["so that", "in order that"]
            }
        },
        
        "relative_pronouns": {
            "who": "for people (subject/object)",
            "whom": "for people (object only, formal)",
            "which": "for things/animals",
            "that": "for people or things (defining clauses)",
            "whose": "for possession",
            "where": "for places",
            "when": "for time"
        },
        
        "conditionals": {
            "type_0": "If + present simple, present simple (general truths)",
            "type_1": "If + present simple, will + infinitive (real possibility)",
            "type_2": "If + past simple, would + infinitive (unreal present)",
            "type_3": "If + past perfect, would have + past participle (unreal past)"
        },
        
        "comparatives": {
            "one_syllable": "add -er/-est (tall -> taller -> tallest)",
            "two_syllables_y": "change y to i + er/est (happy -> happier -> happiest)",
            "two_plus_syllables": "use more/most (expensive -> more expensive -> most expensive)",
            "irregular": {
                "good": "better, best",
                "bad": "worse, worst",
                "far": "farther/further, farthest/furthest",
                "little": "less, least",
                "much/many": "more, most"
            }
        }
    }
    
    result = patterns.get(pattern_type, {})
    
    if not result:
        return {
            "error": f"Pattern type '{pattern_type}' not found",
            "available_types": list(patterns.keys())
        }
    
    return {
        "pattern_type": pattern_type,
        "information": result
    }


@tool
def collocation_check(word: str, context: str = "verb") -> dict:
    """
    Check common collocations for a given word in TOEIC contexts.
    
    Args:
        word: The word to check collocations for
        context: Type of collocation - "verb", "adjective", "noun", "preposition"
    
    Returns:
        Dictionary with common collocations and examples
    """
    
    # Common TOEIC collocations database
    collocations_db = {
        "make": {
            "verb": ["make a decision", "make progress", "make an effort", "make a mistake", 
                    "make an appointment", "make a profit", "make arrangements", "make a reservation"],
            "examples": "We need to make a decision by Friday."
        },
        "take": {
            "verb": ["take action", "take advantage", "take place", "take responsibility",
                    "take a break", "take notes", "take part in", "take effect"],
            "examples": "The meeting will take place in the conference room."
        },
        "do": {
            "verb": ["do business", "do research", "do homework", "do your best",
                    "do damage", "do a favor", "do work"],
            "examples": "We do business with international clients."
        },
        "achieve": {
            "verb": ["achieve goals", "achieve success", "achieve results", "achieve objectives"],
            "examples": "The team achieved excellent results this quarter."
        },
        "reach": {
            "verb": ["reach an agreement", "reach a decision", "reach a conclusion", 
                    "reach a target", "reach a compromise"],
            "examples": "We reached an agreement after long negotiations."
        },
        "conduct": {
            "verb": ["conduct research", "conduct a survey", "conduct an interview", 
                    "conduct business", "conduct a meeting"],
            "examples": "We will conduct a survey next month."
        },
        "raise": {
            "verb": ["raise awareness", "raise funds", "raise concerns", "raise questions",
                    "raise standards", "raise prices"],
            "examples": "The company decided to raise prices by 10%."
        },
        "highly": {
            "adjective": ["highly qualified", "highly recommended", "highly effective",
                         "highly skilled", "highly successful", "highly competitive"],
            "examples": "She is a highly qualified candidate."
        },
        "strong": {
            "adjective": ["strong background", "strong candidate", "strong performance",
                         "strong relationship", "strong demand"],
            "examples": "There is strong demand for this product."
        }
    }
    
    word_lower = word.lower()
    result = collocations_db.get(word_lower, {})
    
    if not result:
        return {
            "word": word,
            "message": "No common collocations found in database",
            "suggestion": "This word may not be commonly tested in TOEIC collocations"
        }
    
    return {
        "word": word,
        "context": context,
        "collocations": result.get(context, result.get("verb", [])),
        "examples": result.get("examples", "")
    }


@tool  
def verb_form_analyzer(sentence: str, blank_position: str = "unknown") -> dict:
    """
    Analyze verb form requirements in a sentence for TOEIC Part 5.
    Helps determine if the blank needs: base form, to-inf, V-ing, past tense, etc.
    
    Args:
        sentence: The sentence with blank
        blank_position: Where the blank is - "after_modal", "after_to", "after_preposition", etc.
    
    Returns:
        Dictionary with verb form analysis and rules
    """
    
    rules = {
        "after_modal": {
            "modals": ["can", "could", "may", "might", "must", "shall", "should", "will", "would"],
            "form_needed": "base form (infinitive without 'to')",
            "example": "She can speak three languages.",
            "note": "Modal verbs are always followed by base form"
        },
        "after_to": {
            "form_needed": "base form (to-infinitive)",
            "example": "He wants to work abroad.",
            "note": "'to' is followed by base form of verb"
        },
        "after_preposition": {
            "form_needed": "V-ing (gerund)",
            "example": "She is good at solving problems.",
            "note": "Prepositions are followed by gerunds (V-ing)"
        },
        "subject_position": {
            "form_needed": "V-ing (gerund) or to-infinitive",
            "example": "Swimming is good exercise. / To work hard is important.",
            "note": "Gerund is more common as subject"
        },
        "after_verb": {
            "form_needed": "depends on the main verb",
            "check": "Use grammar_pattern_lookup for specific verbs",
            "example": "enjoy + V-ing, want + to-inf"
        },
        "main_verb": {
            "form_needed": "conjugated form (matches tense and subject)",
            "tenses": {
                "present_simple": "base form or base+s",
                "past_simple": "V2 or V-ed",
                "present_continuous": "am/is/are + V-ing",
                "past_continuous": "was/were + V-ing",
                "present_perfect": "have/has + V3",
                "past_perfect": "had + V3",
                "future": "will + base form"
            }
        }
    }
    
    # Analyze blank position if specified
    position_info = rules.get(blank_position, {})
    
    # Check for time markers in sentence
    time_markers = {
        "present": ["now", "today", "currently", "usually", "always", "often"],
        "past": ["yesterday", "last", "ago", "in 2020", "previously"],
        "future": ["tomorrow", "next", "will", "soon", "later"],
        "present_perfect": ["already", "yet", "just", "recently", "so far", "since", "for"]
    }
    
    detected_tense = "unknown"
    sentence_lower = sentence.lower()
    for tense, markers in time_markers.items():
        if any(marker in sentence_lower for marker in markers):
            detected_tense = tense
            break
    
    return {
        "sentence": sentence,
        "blank_position": blank_position,
        "position_rules": position_info,
        "detected_tense_hint": detected_tense,
        "all_rules": rules
    }