import re
import time

import spacy
import spacy_transformers

import newspaper
from newspaper import Article

from tqdm import tqdm

class CustomNer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        
    # Parse user query and extract key entity
    def parse_query(self, query):
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "PERSON", "ORG"]]
        return entities or None
    
    def parse_article(self, article, max_retries=3):
        for attempt in range(max_retries):
            try:
                article.download()
                article.parse()
                if article.text:
                    return article.text
                else:
                    print(f"Attempt {attempt + 1}: Article text is empty. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error downloading or parsing article: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(2)
        
        print("Failed to download and parse the article after multiple attempts.")
        return ""

    
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                    }
                )
        return entities
    
    def get_regex_matching_patterns_brief(self, ent1, ent2):
        """
        Returns a dictionary of regex patterns for extracting relationships,
        tailored to the entity types of ent1 and ent2.
        """
        return {
            "LOCATED_IN": re.compile(r"\b(located|lives|in|situated|based)\b") if ent2["label"] == "GPE" else None,
            "WORKS_FOR": re.compile(r"\b(works|employed|for|member of)\b"),
            "PART_OF": re.compile(r"\b(part of|belong to|within)\b") if ent2["label"] in ("ORG", "GPE") else None,
            "IS_RELATED_TO": re.compile(r"\b(related to|connected to)\b"),
            "IS_AFFILIATED_WITH": re.compile(r"\b(is|are|a|an)\b") if ent1["label"] == "PERSON" and ent2["label"] == "ORG" else None,
            "BORN_IN": re.compile(r"\b(born in)\b") if ent1["label"] == "PERSON" and ent2["label"] == "GPE" else None,
            "FOUNDED_BY": re.compile(r"\b(founded by)\b") if ent1["label"] == "ORG" and ent2["label"] == "PERSON" else None,
            "HEADQUARTERED_IN": re.compile(r"\b(headquartered in)\b") if ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
            "HELD_IN": re.compile(r"\b(held in)\b") if ent1["label"] == "EVENT" and ent2["label"] == "GPE" else None,
            "RELEASED_IN": re.compile(r"\b(released in)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "DATE" else None,
        }
    
    def get_regex_matching_patterns(self, ent1, ent2):
        """
        Returns a highly detailed dictionary of regex patterns for extracting relationships,
        tailored to the entity types of ent1 and ent2.
        """
        patterns = {
            # General Relationships
            "MET_WITH": re.compile(r"\b(met|meet|saw|encountered|chatted with|spoke to)\b"),
            "LOCATED_IN": re.compile(r"\b(located|lives|resides|in|situated in|based in)\b") if ent2["label"] == "GPE" else None,
            "WORKS_FOR": re.compile(r"\b(works|employed|for|is employed by|is a member of)\b"),
            "PART_OF": re.compile(r"\b(part of|belong to|within|is a component of|is a subset of)\b") if ent2["label"] == "ORG" or ent2["label"] == "GPE" else None,
            "CONTAINS": re.compile(r"\b(contains|includes|holds|encompasses|comprises)\b") if ent1["label"] == "GPE" and ent2["label"] == "GPE" or ent1["label"] == "ORG" and ent2["label"] == "PRODUCT" or ent1["label"] == "GPE" and ent2["label"] == "ORG" else None,
            "SERVES": re.compile(r"\b(serves|provides|offers|delivers|supplies)\b") if ent1["label"] == "ORG" and ent2["label"] == "PRODUCT" or ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
            "LOCATED_AT": re.compile(r"\b(located at|at|situated at|found at|is based at)\b") if ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
            "IS_RELATED_TO": re.compile(r"\b(is related to|connected to|has ties with|shares a connection with)\b"),
    
            # Person-Specific Relationships
            "IS_AFFILIATED_WITH": re.compile(r"\b(is|are|a|an)\b") if ent1["label"] == "PERSON" and ent2["label"] == "ORG" else None,
            "IS_MARRIED_TO": re.compile(r"\b(married|spouse|partner|husband|wife)\b") if ent1["label"] == "PERSON" and ent2["label"] == "PERSON" else None,
            "IS_RELATED_TO_FAMILY": re.compile(r"\b(child|son|daughter|parent|father|mother|sibling|brother|sister)\b") if ent1["label"] == "PERSON" and ent2["label"] == "PERSON" else None,
            "BORN_IN": re.compile(r"\b(born in|was born in)\b") if ent1["label"] == "PERSON" and ent2["label"] == "GPE" else None,
            "DIED_IN": re.compile(r"\b(died in|passed away in)\b") if ent1["label"] == "PERSON" and ent2["label"] == "GPE" else None,
            "EDUCATED_AT": re.compile(r"\b(educated at|studied at|attended|graduated from)\b") if ent1["label"] == "PERSON" and ent2["label"] == "ORG" else None,
            "AWARDED_TO": re.compile(r"\b(awarded to|received|won|honored with)\b") if ent1["label"] == "ORG" and ent2["label"] == "PERSON" or ent1["label"] == "EVENT" and ent2["label"] == "PERSON" else None,
            "IS_LEADER_OF": re.compile(r"\b(leader of|CEO of|president of|head of|director of)\b") if ent1["label"] == "PERSON" and ent2["label"] == "ORG" else None,
            "IS_MEMBER_OF": re.compile(r"\b(member of|part of|joined|is a member of)\b") if ent1["label"] == "PERSON" and ent2["label"] == "ORG" else None,
            "CREATED": re.compile(r"\b(created|made|developed|invented|authored)\b") if ent1["label"] == "PERSON" and ent2["label"] == "PRODUCT" else None,
            "DIRECTED_BY": re.compile(r"\b(directed by|by|filmed by|produced by)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "PERSON" else None,
    
            # Organization-Specific Relationships
            "FOUNDED_BY": re.compile(r"\b(founded by|established by|created by|started by)\b") if ent1["label"] == "ORG" and ent2["label"] == "PERSON" else None,
            "PRODUCED_BY": re.compile(r"\b(produced by|manufactured by|made by|built by)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "ORG" else None,
            "INVESTED_IN": re.compile(r"\b(invested in|funding|financed|backed)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
            "PUBLISHED_BY": re.compile(r"\b(published by|released by|distributed by)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "ORG" else None,
            "HEADQUARTERED_IN": re.compile(r"\b(headquartered in|is based in|has its headquarters in)\b") if ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
            "SUBSIDIARY_OF": re.compile(r"\b(subsidiary of|is a subsidiary of|owned by)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
    
            # Event-Specific Relationships
            "HELD_IN": re.compile(r"\b(held in|took place in|occurred in)\b") if ent1["label"] == "EVENT" and ent2["label"] == "GPE" else None,
            "PARTICIPATED_IN": re.compile(r"\b(participated in|attended|was present at)\b") if ent1["label"] == "PERSON" and ent2["label"] == "EVENT" else None,
            "ORGANIZED_BY": re.compile(r"\b(organized by|hosted by|sponsored by)\b") if ent1["label"] == "EVENT" and ent2["label"] == "ORG" else None,
    
            # Product-Specific Relationships
            "HAS_DATE": re.compile(r"\b(is|are|has|have|was|were)\b") if ent2["label"] == "DATE" else None,
            "HAS_NUMBER": re.compile(r"\b(is|are|has|have|was|were)\b") if ent2["label"] == "CARDINAL" else None,
            "HAS_PRODUCT": re.compile(r"\b(is|are|has|have|includes|contains)\b") if ent2["label"] == "PRODUCT" else None,
            "RELEASED_IN": re.compile(r"\b(released in|launched in)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "DATE" else None,
            "SOLD_IN": re.compile(r"\b(sold in|available in)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "GPE" else None,
    
            # Date-Specific Relationships
            "OCCURRED_ON": re.compile(r"\b(occurred on|happened on|took place on)\b") if ent1["label"] == "EVENT" and ent2["label"] == "DATE" else None,
            "VALID_UNTIL": re.compile(r"\b(valid until|expires on)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "DATE" else None,
    
            # Cardinal-Specific Relationships
            "MEASURED_IN": re.compile(r"\b(measured in|counts|amounts to)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "CARDINAL" else None,
            "POPULATION_OF": re.compile(r"\b(population of|has a population of)\b") if ent1["label"] == "GPE" and ent2["label"] == "CARDINAL" else None,
            "NUMBER_OF": re.compile(r"\b(number of|consists of|contains)\b") if ent1["label"] == "ORG" and ent2["label"] == "CARDINAL" or ent1["label"] == "PRODUCT" and ent2["label"] == "CARDINAL" else None,
    
            # GPE-Specific Relationships
            "CAPITAL_OF": re.compile(r"\b(capital of|is the capital of)\b") if ent1["label"] == "GPE" and ent2["label"] == "GPE" else None,
            "BORDERED_BY": re.compile(r"\b(bordered by|shares a border with|adjacent to)\b") if ent1["label"] == "GPE" and ent2["label"] == "GPE" else None,
            "IS_IN_REGION": re.compile(r"\b(is in|is part of|located in)\b") if ent1["label"] == "GPE" and ent2["label"] == "GPE" else None,
    
            # Event to Event relationships
            "PRECEDED_BY": re.compile(r"\b(preceded by|occurred before)\b") if ent1["label"] == "EVENT" and ent2["label"] == "EVENT" else None,
            "FOLLOWED_BY": re.compile(r"\b(followed by|occurred after)\b") if ent1["label"] == "EVENT" and ent2["label"] == "EVENT" else None,
            "CAUSED_BY": re.compile(r"\b(caused by|resulted from)\b") if ent1["label"] == "EVENT" and ent2["label"] == "EVENT" else None,
            "LED_TO": re.compile(r"\b(led to|resulted in)\b") if ent1["label"] == "EVENT" and ent2["label"] == "EVENT" else None,
    
            # Product to Product relationships
            "IS_A_TYPE_OF": re.compile(r"\b(is a type of|is a kind of|is a variation of)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "PRODUCT" else None,
            "REQUIRES": re.compile(r"\b(requires|needs|uses)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "PRODUCT" else None,
            "COMPATIBLE_WITH": re.compile(r"\b(compatible with|works with)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "PRODUCT" else None,
            "REPLACES": re.compile(r"\b(replaces|substitutes|is a replacement for)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "PRODUCT" else None,
    
            # ORG to ORG Relationships
            "MERGED_WITH": re.compile(r"\b(merged with|acquired by)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
            "PARTNERED_WITH": re.compile(r"\b(partnered with|collaborated with)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
            "COMPETES_WITH": re.compile(r"\b(competes with|rivals)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
            "SUPPLIES": re.compile(r"\b(supplies|provides)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
            "DISTRIBUTES": re.compile(r"\b(distributes|sells)\b") if ent1["label"] == "ORG" and ent2["label"] == "ORG" else None,
    
            # Person to Event Relationships
            "ATTENDED": re.compile(r"\b(attended|participated in)\b") if ent1["label"] == "PERSON" and ent2["label"] == "EVENT" else None,
            "SPOKE_AT": re.compile(r"\b(spoke at|gave a speech at)\b") if ent1["label"] == "PERSON" and ent2["label"] == "EVENT" else None,
            "PERFORMED_AT": re.compile(r"\b(performed at|played at)\b") if ent1["label"] == "PERSON" and ent2["label"] == "EVENT" else None,
    
            # Person to Product Relationships
            "USES": re.compile(r"\b(uses|utilizes|employs)\b") if ent1["label"] == "PERSON" and ent2["label"] == "PRODUCT" else None,
            "OWNS": re.compile(r"\b(owns|possesses)\b") if ent1["label"] == "PERSON" and ent2["label"] == "PRODUCT" else None,
            "REVIEWED": re.compile(r"\b(reviewed|rated)\b") if ent1["label"] == "PERSON" and ent2["label"] == "PRODUCT" else None,
    
            # Product to GPE Relationships
            "SHIPPED_TO": re.compile(r"\b(shipped to|delivered to)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "GPE" else None,
            "MANUFACTURED_IN": re.compile(r"\b(manufactured in|produced in)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "GPE" else None,
            "AVAILABLE_IN": re.compile(r"\b(available in|sold in)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "GPE" else None,
    
            # Org to GPE Relationships
            "OPERATES_IN": re.compile(r"\b(operates in|has branches in)\b") if ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
            "IS_BASED_IN": re.compile(r"\b(is based in|located in)\b") if ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
            "HAS_OFFICES_IN": re.compile(r"\b(has offices in|maintains a presence in)\b") if ent1["label"] == "ORG" and ent2["label"] == "GPE" else None,
    
            # Event to GPE Relationships
            "HELD_IN_GPE": re.compile(r"\b(held in|took place in)\b") if ent1["label"] == "EVENT" and ent2["label"] == "GPE" else None,
    
            # Event to Date Relationships
            "OCCURRED_ON_DATE": re.compile(r"\b(occurred on|happened on|took place on)\b") if ent1["label"] == "EVENT" and ent2["label"] == "DATE" else None,
    
            # Product to Date Relationships
            "RELEASED_ON": re.compile(r"\b(released on|launched on)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "DATE" else None,
            "EXPIRES_ON": re.compile(r"\b(expires on|valid until)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "DATE" else None,
    
            # GPE to CARDINAL Relationships
            "POPULATION_CARDINAL": re.compile(r"\b(population of|has a population of)\b") if ent1["label"] == "GPE" and ent2["label"] == "CARDINAL" else None,
    
            # ORG to CARDINAL Relationships
            "EMPLOYEES_CARDINAL": re.compile(r"\b(employees|staff|workforce)\b") if ent1["label"] == "ORG" and ent2["label"] == "CARDINAL" else None,
    
            # Product to CARDINAL Relationships
            "COST_CARDINAL": re.compile(r"\b(costs|priced at|valued at)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "CARDINAL" else None,
            "QUANTITY_CARDINAL": re.compile(r"\b(quantity|amount|number)\b") if ent1["label"] == "PRODUCT" and ent2["label"] == "CARDINAL" else None,
        }
        
        return patterns
    
    def extract_relationships(self, text, entities):
    
        relationships = []
        text_lower = text.lower()
        entity_positions = {}  # Store entity positions to avoid redundant searches
    
        # Pre-compute entity positions
        for ent in entities:
            ent_lower = ent["text"].lower()
            start = text_lower.find(ent_lower)
            if start != -1:
                entity_positions[ent["text"]] = (start, start + len(ent["text"]))
    
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i + 1:], i + 1):
                if ent1["text"] not in entity_positions or ent2["text"] not in entity_positions:
                    continue
    
                ent1_start, ent1_end = entity_positions[ent1["text"]]
                ent2_start, ent2_end = entity_positions[ent2["text"]]
    
                if ent2_start > ent1_end:
                    between_text = text_lower[ent1_end:ent2_start]
    
                    patterns = self.get_regex_matching_patterns(ent1, ent2)
    
                    for rel_type, pattern in patterns.items():
                        if pattern and pattern.search(between_text):
                            neo4j_relationship = {
                                "entity1": {"text": ent1["text"], "label": ent1["label"]},
                                "relation": "IS_RELATED_TO" if rel_type.startswith("IS_RELATED_TO_") else rel_type,
                                "entity2": {"text": ent2["text"], "label": ent2["label"]},
                            }
                            relationships.append(neo4j_relationship)
    
        return relationships

    