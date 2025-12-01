"""
Exploration Script: Understanding Probabilities in NTSB Data

This script helps us understand:
1. What events occur in our data
2. How often events follow each other (transitions)
3. What probabilities we can calculate

Run this to see what we're working with before building the full system.
"""

import json
from collections import Counter, defaultdict

# Load the data
print("=" * 70)
print("EXPLORING NTSB DATA FOR BAYESIAN NETWORK")
print("=" * 70)

with open('Data/refined_dataset.json', 'r') as f:
    refined_data = json.load(f)

print(f"\n📊 Total incidents in database: {len(refined_data)}")

# ============ PART 1: UNDERSTAND EVENT SEQUENCES ============
print("\n" + "=" * 70)
print("PART 1: EXPLORING EVENT SEQUENCES")
print("=" * 70)

# Let's look at a few examples first
print("\n🔍 Example 1: Looking at a single incident in detail")
print("-" * 70)

# Get first incident
first_ev_id = list(refined_data.keys())[0]
first_incident = refined_data[first_ev_id]

print(f"Incident ID: {first_ev_id}")
print(f"\nSequence of events:")
for i, event in enumerate(first_incident.get('sequence_of_events', []), 1):
    print(f"  {i}. {event['Occurrence_Description']}")
    print(f"     - Occurrence Code: {event['Occurrence_Code']}")
    print(f"     - Phase: {event['phase_no']}")
    print(f"     - Defining Event: {'Yes' if event.get('Defining_ev') == 1 else 'No'}")

# ============ PART 2: COUNT ALL EVENTS ============
print("\n" + "=" * 70)
print("PART 2: COUNTING EVENTS ACROSS ALL INCIDENTS")
print("=" * 70)

event_counter = Counter()
incidents_with_sequences = 0
total_events = 0

for ev_id, incident in refined_data.items():
    sequence = incident.get('sequence_of_events', [])
    if sequence:
        incidents_with_sequences += 1
        for event in sequence:
            event_desc = event.get('Occurrence_Description', 'Unknown')
            event_counter[event_desc] += 1
            total_events += 1

print(f"\n📈 Statistics:")
print(f"  - Incidents with event sequences: {incidents_with_sequences}")
print(f"  - Total events recorded: {total_events}")
print(f"  - Unique event types: {len(event_counter)}")

print(f"\n🏆 Top 10 Most Common Events:")
for i, (event, count) in enumerate(event_counter.most_common(10), 1):
    probability = count / incidents_with_sequences
    print(f"  {i:2d}. {event}")
    print(f"      Count: {count}, P(event) = {probability:.4f} ({probability*100:.2f}%)")

# ============ PART 3: CALCULATE TRANSITIONS ============
print("\n" + "=" * 70)
print("PART 3: CALCULATING TRANSITION PROBABILITIES P(B|A)")
print("=" * 70)

print("\n📚 What is a transition probability?")
print("  P(Event B | Event A) = 'Probability of B happening, given A happened'")
print("  Formula: P(B|A) = Count(A→B) / Count(A appears)")
print()

# Count transitions: Event A → Event B
transition_counts = defaultdict(Counter)

for ev_id, incident in refined_data.items():
    sequence = incident.get('sequence_of_events', [])
    
    # Look at each pair of consecutive events
    for i in range(len(sequence) - 1):
        event_a = sequence[i].get('Occurrence_Description', 'Unknown')
        event_b = sequence[i+1].get('Occurrence_Description', 'Unknown')
        
        # Count this transition
        transition_counts[event_a][event_b] += 1

print(f"✅ Transitions calculated!")
print(f"  - Events that lead to other events: {len(transition_counts)}")

# Show some examples
print(f"\n🔗 Example Transitions (for most common events):")

# Get top 5 most common events
top_events = [event for event, count in event_counter.most_common(5)]

for event_a in top_events[:3]:  # Show first 3 to keep output manageable
    print(f"\n  When '{event_a}' occurs:")
    print(f"    (appeared {event_counter[event_a]} times total)")
    
    # Get what follows this event
    next_events = transition_counts[event_a]
    if next_events:
        print(f"    It is followed by:")
        for event_b, count in next_events.most_common(5):  # Top 5 followers
            # Calculate conditional probability
            prob = count / event_counter[event_a]
            print(f"      → {event_b}")
            print(f"         P(next|current) = {count}/{event_counter[event_a]} = {prob:.4f} ({prob*100:.1f}%)")
    else:
        print(f"    (This was the final event in sequences)")

# ============ PART 4: UNDERSTANDING THE WEIGHTED PRIOR CONCEPT ============
print("\n" + "=" * 70)
print("PART 4: DEMONSTRATION - WEIGHTED PRIORS")
print("=" * 70)

print("\n💡 Key Concept: Weighted Priors Based on Similarity")
print("-" * 70)
print("Instead of treating all incidents equally, we weight them by")
print("how similar they are to the current query.")
print()
print("Example scenario:")
print("  Query: 'Engine failure during takeoff'")
print()
print("  Similar incidents found (by cosine similarity):")
print("    Incident A: similarity = 0.92  → had 'Loss of power'")
print("    Incident B: similarity = 0.87  → had 'Loss of power'")  
print("    Incident C: similarity = 0.81  → NO 'Loss of power'")
print("    Incident D: similarity = 0.75  → had 'Loss of power'")
print()
print("  Weighted probability of 'Loss of power':")
print("    P = (0.92×1 + 0.87×1 + 0.81×0 + 0.75×1) / (0.92+0.87+0.81+0.75)")
print("    P = 2.54 / 3.35 = 0.758 (75.8%)")
print()
print("  Compare to unweighted (simple average):")
print("    P = (1 + 1 + 0 + 1) / 4 = 0.75 (75%)")
print()
print("  The weighted version gives MORE influence to the most similar incidents!")

# ============ SUMMARY ============
print("\n" + "=" * 70)
print("SUMMARY: WHAT WE CAN CALCULATE")
print("=" * 70)

print("\n✅ From this data, we can pre-compute:")
print("  1. P(Event) - How often each event occurs")
print(f"     Example: P('{top_events[0]}') = {event_counter[top_events[0]]/incidents_with_sequences:.4f}")
print()
print("  2. P(Event B | Event A) - Transition probabilities")
if transition_counts[top_events[0]]:
    next_event = transition_counts[top_events[0]].most_common(1)[0]
    prob = next_event[1] / event_counter[top_events[0]]
    print(f"     Example: P('{next_event[0]}' | '{top_events[0]}') = {prob:.4f}")
print()
print("  3. P(Event | Query) - Weighted by similarity to query")
print("     Example: Use cosine similarity as weights (as shown above)")
print()
print("  4. P(sequence) - Probability of entire event chain")
print("     Example: P(A→B→C) = P(A) × P(B|A) × P(C|B)")

print("\n" + "=" * 70)
print("NEXT STEP: Build script to pre-compute all these probabilities")
print("           and store them with each incident for fast lookup!")
print("=" * 70)
























