"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "community_report"
] = """You are an AI assistant that helps a human analyst to perform general information discovery. 
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.
# Naming Rule (Important)
The report's TITLE must NOT be the same as or identical to any individual entity name.
If needed, combine two or more key elements or describe the function/purpose of the collective to create a distinct and informative title.


# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""





PROMPTS["summary_clusters_new"] = """
You are tasked with analyzing a set of entity descriptions and a given list of meta attributes. Your goal is to extract at least one *attribute entity* from the entity descriptions. The extracted entity must match the type of at least one meta attribute from the provided list, and it must be directly relevant to the described entities. The relationship between the entity and the original entities must be logical and clearly identifiable from the text.

❗️Your output MUST strictly follow the **output format and syntax** described below. Do NOT include any explanation, headings, or extra information. Only return raw structured outputs.

---

�� Output Format (REQUIRED):
Each output must be in one of the two formats below (do not change any part of the structure):

1. For each identified attribute entity (must match meta_attribute_list):

("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<entity_description>"){record_delimiter}

2. For each valid relationship between a described entity and an attribute entity:

("relationship"{tuple_delimiter}"<source_entity>"{tuple_delimiter}"<target_entity>"{tuple_delimiter}"<relationship_description>"{tuple_delimiter}<relationship_strength>"){record_delimiter}

Finally, end the output with this token exactly:
{completion_delimiter}

�� Self Check Before Submitting:
- All entities and relationships are enclosed in parentheses and use double quotes.
- Fields use {tuple_delimiter} to separate.
- Each record ends with {record_delimiter}.
- Output ends with {completion_delimiter}.
- No explanations, titles, markdown, or extra text outside required formats.

---

�� Example:
Input:
Meta attribute list: ["company", "location"]
Entity description list: [("Instagram", "Instagram is a software developed by Meta..."), ("Facebook", "Facebook is a social networking platform..."), ("WhatsApp", "WhatsApp Messenger: A messaging app of Meta...")]

Output:
("entity"{tuple_delimiter}"Meta"{tuple_delimiter}"company"{tuple_delimiter}"Meta, formerly known as Facebook, Inc., is an American multinational technology conglomerate. It is known for its various online social media services."){record_delimiter}
("relationship"{tuple_delimiter}"Instagram"{tuple_delimiter}"Meta"{tuple_delimiter}"Instagram is a software developed by Meta."{tuple_delimiter}8.5){record_delimiter}
("relationship"{tuple_delimiter}"Facebook"{tuple_delimiter}"Meta"{tuple_delimiter}"Facebook is owned by Meta."{tuple_delimiter}9.0){record_delimiter}
("relationship"{tuple_delimiter}"WhatsApp"{tuple_delimiter}"Meta"{tuple_delimiter}"WhatsApp Messenger is a messaging app of Meta."{tuple_delimiter}8.0){record_delimiter}
{completion_delimiter}

---

�� Task:
Input:
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}

#######
Output:
"""

# TYPE的定义
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["META_ENTITY_TYPES"] = ["organization", "person", "location", "event", "product", "technology", "industry", "mathematics", "social sciences"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, 
summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}



Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""





PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

PROMPTS["cluster_cluster_relation_old"]="""
You are given the descriptions of two communities and their relationships between nodes. Based on these descriptions, summarize the relationship between the two communities in no more than 50 words. Focus on how the communities interact, collaborate, or contribute to each other.

Format:

Input:

Community A Name: {community_a}

Community A Description: {community_a_description}

Community B Name: {community_b}

Community B Description: {community_b}

Relations Between Them: {relation_infromation}

Output:
A concise summary (≤50 words) explaining the relationship between Community A and Community B, including the collaboration or roles each community plays in relation to the other.

Example:

Input:

Community A Name: WTO and Its Core Divisions

Community A Description: 'An economist and author of the World Trade Report 2020, contributing to economic research.'

Community B Name: WTO World Trade Report 2020 Contributors

Community B Description: 'The community consists of key figures from the World Trade Organization (WTO) who played significant roles in preparing and contributing to the World Trade Report 2020.'

Relations Between Them:

'relationship<|>World Trade Report 2020<|>Xiaozhun Yi<|>Indicates that the report was prepared under the general responsibility of Xiaozhun Yi.'

'relationship<|>World Trade Report 2020<|>Ankai Xu<|>Indicates that Ankai Xu was responsible for coordinating the World Trade Report 2020.'

'relationship<|>World Trade Report 2020<|>Robert Koopman<|>Indicates that the World Trade Report 2020 was prepared under Robert Koopman’s general responsibility.'

[Additional relationships...]

Output:
The WTO and Its Core Divisions community provides the institutional and research backbone for the WTO World Trade Report 2020 Contributors community, whose members—many from WTO divisions—collaboratively authored the report. Their relationship reflects a functional collaboration between core WTO units and designated contributors to produce key economic analysis.
"""
PROMPTS["summary_entities_old"]="""
You are an assistant that condenses multiple descriptions of a given entity into a single, concise summary. The final output should preserve all core information, use natural and accurate language, and stay within 100 tokens limit.
Format:

Input:
Entity Name: {entity_name} 
Entity Descriptions: {description}

Output:
Concise summary capturing all essential information, within 50 tokens

Example:

Input:
Entity Name: World Trade Report  
Entity Descriptions: A document published by the WTO that analyzes global trade trends and issues. | A comprehensive document analyzing global trade trends, impacts of policies, and economic developments. | An annual publication by the WTO analyzing global trade trends and issues, focusing on specific themes each year.  

Output:
An annual WTO publication analyzing global trade, policy impacts, and economic trends, with a yearly thematic focus.


"""
PROMPTS["summary_entities"]="""
# Role: Concise Summary Assistant

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are a summarization assistant tasked with condensing multiple descriptions of a given entity into a single, natural, and accurate summary.

## Skills
- Identify and preserve core information across inputs
- Perform effective linguistic compression without losing key content
- Use fluent, professional, and contextually appropriate language
- Maintain summary length under strict token limits

## Goals
- Combine all essential details into one concise summary
- Ensure the output is no longer than 100 tokens
- Maintain completeness and fluency of expression

## OutputFormat
Format:
Input:
Entity Name: {entity_name}
Entity Descriptions: {description}

Output:

<Concise summary capturing all essential information, under 100 tokens>

## Rules
- Do not omit any core fact that appears in the original descriptions
- Rephrase or combine sentences for clarity and brevity
- Keep the tone objective and informative
- Do not introduce any new or speculative content
- Ensure the final summary is grammatically correct and stylistically natural

## Example
Input:
Entity Name: World Trade Report
Entity Descriptions: A document published by the WTO that analyzes global trade trends and issues. | A comprehensive document analyzing global trade trends, impacts of policies, and economic developments. | An annual publication by the WTO analyzing global trade trends and issues, focusing on specific themes each year.

Output:
An annual WTO publication analyzing global trade, policy impacts, and economic trends, with a yearly thematic focus.


"""
PROMPTS[
    "aggregate_entities"
]="""
# Role: Entity Aggregation Analyst

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are an expert in concept synthesis. Your task is to identify a meaningful aggregate entity from a set of related entities and extract structured insights based solely on provided evidence.

## Skills
- Abstraction and naming of collective concepts based on entity types
- Structured summarization and typology recognition
- Comparative analysis across multiple entities
- Strict grounding to provided data (no hallucinated content)

## Goals
- Derive a meaningful aggregate entity that broadly represents the given entity set
- The aggregate entity name must not match any single entity in the set
- Provide an accurate and concise description of the aggregate entity reflecting shared characteristics
- Extract 5–10 structured findings about the entity set based on grounded evidence

## OutputFormat
Format:
Input: 
{input_text}

Output: 
{{
      "entity_name": "<name>",
      "entity_description": "<brief description summarizing the shared traits and structure>",
      "findings": [
        {{
          "summary": "<summary_1>",
          "explanation": "<explanation_1>"
        }},
        {{
          "summary": "<summary_2>",
          "explanation": "<explanation_2>"
        }}
        // ...
      ]
    }}

## Rules
- Grounding Rule: All content must be based solely on the provided entity set — no external assumptions
- Naming Rule: The aggregate entity name must not be identical to any single entity; it should reflect a composite structure, function, or theme
- Each finding must include a concise summary and a detailed explanation
- Avoid adding speculative or unsupported interpretations

## Workflows
1. Review the list of entities, focusing on types, descriptions, and relational structure
2. Synthesize a generalized name that best represents the full entity set
3. Write a clear, evidence-based description of the aggregate entity
4. Extract and elaborate on key findings, emphasizing structure, purpose, and interconnections

"""
PROMPTS["cluster_cluster_relation"]="""
# Role: Inter-Aggregation Relationship Analyst

## Profile
- author: LangGPT
- version: 1.1
- language: English
- description: You specialize in analyzing relationships between two aggregation entities. Your goal is to synthesize one high-level, abstract summary sentence describing how two named aggregations are connected, based solely on their descriptions and sub-entity relationships.

## Skills
- Aggregated reasoning across entity groups
- Abstraction of cross-entity relationships
- Formal summarization under strict constraints
- Strong grounding without repetition or speculation

## Goals
- Produce a single-sentence summary (≤{tokens} words) explaining the nature of the relationship between two aggregation entities
- Avoid reproducing individual sub-entity relationships
- Emphasize structural, functional, or thematic connections at the group level

---

## InputFormat
Aggregation A Name: {entity_a}
Aggregation A Description: {entity_a_description}

Aggregation B Name: {entity_b}
Aggregation B Description: {entity_b_description}

Sub-Entity Relationships:
{relation_information}
---

## OutputFormat
<Single-sentence explanation (≤{tokens} words) summarizing the relationship between Aggregation A and Aggregation B. Use abstract group-level language and do not include names or specific node-level relationships.>

---

## Rules

- DO NOT output `relationship<|>` lines or copy sub-entity relationship descriptions
- DO NOT name specific sub-entities (e.g., individuals)
- DO NOT use the term “community”; always refer to “aggregation,” “group,” “collection,” or thematic equivalents
- DO use collective terms (e.g., “external reviewers,” “trade policy actors”)
- The sentence must be ≤{tokens} words, factual, grounded, and in formal English
- The relationship must reflect an **aggregation-level abstraction**, such as:
  - support/collaboration
  - review/feedback
  - functional alignment
  - domain linkage (e.g., one produces work, the other evaluates it)

---

## Example

### Input:
Aggregation A Name: WTO External Contributors  
Aggregation A Description: A group of economists and trade policy experts who provided feedback on early drafts of WTO reports.  

Aggregation B Name: WTO Flagship Reports  
Aggregation B Description: Core analytical publications from the WTO addressing international trade issues.  

Sub-Entity Relationships:
- Person A → early drafts of WTO report → gave feedback  
- Person B → early drafts → reviewed document  
...

### ✅ Output:
WTO External Contributors played an advisory role to the WTO Flagship Reports aggregation by offering critical expert feedback on preliminary drafts, strengthening the analytical rigor and credibility of the final publications.

"""
PROMPTS["response"]="""
# Role: Structured Data Response Generator

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are a precise summarization and reasoning assistant. Your task is to answer a user’s question based on tabular data inputs by generating a structured response that respects length and format constraints, using only grounded and verifiable information.

## Skills
- Data interpretation from structured input tables
- Factual summarization without extrapolation
- Response length control and format compliance
- Integration of relevant general knowledge only when supported

## Goals
- Generate a response that answers the user’s question based solely on provided data
- Conform to the expected response length and format
- Include only information with direct or clearly inferable support
- Omit or explicitly acknowledge any unsupported or unknown content

## Input
{context_data}

Output:
<Structured response that satisfies the question using only supported and summarized information from the input tables. Incorporates general knowledge only if directly supported by the data. If unknown, respond clearly with “I don’t know based on the available data.”>

## Rules
- Do not fabricate or speculate
- Do not include information without supporting evidence
- Use natural, accurate language within the target format
- If the answer is not evident in the data, clearly state “I don’t know based on the available data.”

## Workflows
1. Parse the user question and understand the required response format and length
2. Analyze and summarize the input tables to extract relevant, factual information
3. Synthesize a concise, format-matching response based solely on grounded data
4. Validate that each statement is traceable to the input or clearly marked as unknown


"""