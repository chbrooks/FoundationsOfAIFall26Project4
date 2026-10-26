### Project 4

#### Due Tues, Nov 10, 11:59pm

### Question 1: Logical Inference

NASA has reached out to us for help in designing the logic for the Mars rover. They would like it to be able to conduct experiments, and only radio back to Earth for help in case of an actual emergency. We've decided to build a rule-based system to implement this.

We have the following predicates (with abbreviations) to use:

- a: battery failed
- b: solar panels failed
- c: base station failed
- d: backup battery failed
- e: send emergency signal
- q: backup switch fails
- w: cannot sync base station with rover
- x: battery sensor not responding
- y: battery will not power motor
- z: no current from panels

NASA has given us the following rules to use:

- If the battery sensor is not responding and the battery will not power the motor, then the battery failed.
- If the battery sensor is not responding, we are not able to sync the base station with the rover.
- If the battery sensor is not responding and there is no current from the panels, then the solar panels have failed.
- If the battery will not power the motor, and we cannot sync the base station with the rover, then the base station has failed.
- If the backup switch fails, and there's no current from the panels, and the battery fails, then the backup battery fails.
- If the backup switch fails, then the battery will not power the motor.
- If battery failed and solar panels failed and base station failed and backup battery failed, then send the emergency signal.
- If we cannot sync the base station with the rover, we are unable to get current from the panels.

Our rover has observed the following facts:

- The battery sensor is not responding.
- The backup switch has failed.

**a)** Show what the initial KB looks like, using the abbreviations for each rule and fact.

**b)** Use forward chaining to show that we should send an emergency signal. For each step, show the rule and fact(s) being matched, and the resulting fact. for example: q, q->y, y.

**c)** Our NASA contact is skeptical; he's heard that AI can make mistakes, and does not believe that there is truly an emergency. We need to prove that this was the right decision. Use backward chaining to show that signaling the emergency was the correct decision. For each step, show the stack and the items to be proved.


### Question 2: Knowledge Graphs vs LLMs. 

In this question, you'll build a knowledge graph of IMDB data and compare its performance to the performance of an LLM. You'll get experience working with NetworkX and Ollama. 

Claude and I have built a great deal of the code for you. You have a few functions to implement, as well as some analysis to do. 
You can use Claude to help you understand the code, but you should implement the answers yourself. 


**Step 1.** Download and unzip these four files. 

https://datasets.imdbws.com/title.basics.tsv.gz
https://datasets.imdbws.com/title.principals.tsv.gz
https://datasets.imdbws.com/name.basics.tsv.gz
https://datasets.imdbws.com/title.ratings.tsv.gz

and install the following packages: networkx requests tqdm

Then install ollama from https://ollama.ai

and pull llama3. 

(Friday's asynchronous lab will go over NetworkX and Ollama)

You can then build the graph with:

```
python kg_factcheck.py --build          # build and cache the graph
```

This will take a little time.

Once it's built, run this in a Python console:
```
import json, networkx as nx
with open("imdb_kg.json") as f:
    G = nx.node_link_graph(json.load(f), directed=True)

# Inspect
print(G.number_of_nodes(), G.number_of_edges())

# Degree distribution — who has the most connections?
degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:100]
for node, deg in degrees:
    print(G.nodes[node].get("name") or G.nodes[node].get("title", node), deg)

```

This will print the 100 nodes with the largest number of edges. 

**Question**  Does that match your intuition about prolific film careers?

**Step 2.2-hop traversal: Common directors.**

Claude has built a KnowledgeGraphChecker class that is able to look for 
particular patterns in the graph. In particular, it can find relationships of 
the form actor A acted in Movie M with person B. It does this in two ways - 
by looking for a CO_STAR edge, and then by explicitly looking for films that are in both of 
their outgoing edges. 

Add a similar method called directed_both(G, film1, film2) that takes as input 
a graph G and two names of films and returns the list of persons who directed BOTH 
films. 

**Step 3. 3-hop traversals and n-hop traversals.**

We would now like to find all actors who appear in a pair of films. Let's call
this actor_bridge. Please add this to kg_factcheck.py

```
def actor_bridges(G, film1_title, film2_title):
    """
    Returns a list of (actor_name, film1_title, film2_title) tuples for
    actors who appeared in both films.
    Use set intersection on the ACTED_IN neighborhood of each film.
    """
    # YOUR CODE HERE
    pass
```
Then extend this to a chain: 
given a list of film titles, find a chain of actors
who connect every consecutive pair —i.e.,actor A appeared in film 1 and film 2, 
actor B appeared in film 2 and film 3, etc.

```
def actor_chain(G, film_titles):
    """
    For each consecutive pair of films, find at least one bridging actor.
    Returns a list of (actor_name, film_i_title, film_{i+1}_title) tuples,
    or None for any pair with no shared actor.
    """
    # YOUR CODE HERE
    pass
```

**Step 4: Benchmarking**

Run the benchmark like so:
```
python kg_factcheck_lab.py --benchmark --llm  
```
It takes a minute to load the graph. 

The benchmark contains 14 claims across three difficulty levels:

- 1-hop:single relation (directed, acted, year, genre)
- 2-hop:path through two relations (co-starred, same director, director of genre)
- 3-hop:three-relation chain (shared cast + genre)

And three answer types:

- TRUE— the claim is verifiable from the graph
- FALSE— the graph contains a contradicting path
- UNKNOWN— the relevant entities are absent from the graph

**Question** 

1. KG accuracy: how many claims did the graph checker get right?
2. LLM accuracy how many did the LLM get right?
3. List any disagreements— cases where KG and LLM gave different verdicts. 
For each disagreement, determine what went wrong:

- KG correct, LLM wrong:Did the LLM hallucinate? (state a confident FALSE/TRUE that isn't supported by training data) What's your evidence?
- LLM correct, KG wrong:Is the fact simply missing from your filtered graph? What would you need to add to fix the KG answer?
- Both wrong: Is this a hard claim that requires external knowledge not representable as a simple path?

**Step 5 More Claims** 

In this step, you should add two additional claims in KnowledgeGraphChecker's claim() method.

 prolific_director: "Has <person> directed more than <n> films in this graph?" (1-hop count query)

 genre_year_range: "Were there any <genre> films released between <year_min> and <year_max>?" (multi-entity filter)
For each:

1. Implement the claim type in the check() method.
2. Write two benchmark entries (one TRUE, one FALSE).
3. Compare KG and LLM answers.

**Step 6: Wrap-up.** 

Answer the following questions. Be specific — reference your actual benchmark results.

1. What did the LLM get wrong, and why? Pick the most interesting LLM error from your benchmark run. Was it a hallucination (confident but fabricated), a calibration failure (wrong confidence), or a knowledge gap (fact not in training data)? What would it take to prevent this error?

2. What did the KG get wrong, and why? Pick one KG failure (FALSE or UNKNOWN on a claim that should be TRUE). Is this a coverage problem (data not loaded), a schema problem (the relation type doesn't exist in your graph), or a traversal problem (the path exists but your code didn't find it)?

3. Multi-hop reasoning and LLMs. Try a 3-hop or chain question in the REPL (sc command) and ask the same question to an LLM (either via Ollama or a web interface). Which system answered correctly? Why are 3-hop questions hard for LLMs? (Hint: think about how attention distributes over a long chain of reasoning, vs. an explicit graph traversal.)

5. KG + LLM: is there a better architecture? Describe (in a few sentences — no implementation needed) a system that uses both a KG and an LLM in tandem. What would each component handle? 
When would you trust the KG over the LLM, and vice versa? This is the core idea behind retrieval-augmented generation (RAG).


### Part 3: Grad students only.

Within the subfield of knowledge representation, there's been a great deal of work done on developing ontologies. An ontology is a set of concepts, categories and relationships between them with respect to a specific domain. Ontologies have been very effective at allowing for complex inference in specific domains, such as manufacturing or medicine.

Please read (this webpage)[https://www.earley.com/insights/role-ontology-and-information-architecture-ai]] and answer the following questions:

1. What kinds of relationships can be captured in an ontology?
2. How does an ontology differ from a taxonomy?
3. What is the difference between an ontology and a knowledge graph?
4. How can an ontology assist an organization that might have multiple AI applications?
5. How can ontologies assist LLMs? 



### Optional Final Exam Replacement Question

The final exam will consist of four questions. If you complete all four, you will get a '+' on your grade.
If you complete 2 or 3, no modifier. 1 or 0 is a '-'.

You may choose to complete this question and have it count as one final exam question.

Extend Step 5 of part 2 to add at least three additional claims to your fact checker. 
For each claim, write a TRUE and a FALSE benchmark and compare the performance
of the KG and the LLM, and cmpare their answers. 
