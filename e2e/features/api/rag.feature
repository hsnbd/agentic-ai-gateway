@api
Feature: Retrieval-augmented generation
  Documents are chunked, embedded, and stored in Redis vector indexes.

  Background:
    Given I use the master key
    And a RAG collection with the documents:
      | title  | content                                                            |
      | paris  | Paris is the capital of France and home of the Eiffel Tower.       |
      | tokyo  | Tokyo is the capital of Japan and has the busiest railway station. |

  Scenario: Semantic search finds the relevant document
    When I search the collection for "Tokyo railway station"
    Then the top search result mentions "Tokyo"

  Scenario: A RAG query answers with its sources
    When I query the collection with "What is the capital of France?"
    Then the response status is 200
    And the RAG answer cites a source mentioning "Paris"

  Scenario: Documents can be removed
    When I delete the "paris" document
    And I search the collection for "Eiffel Tower Paris"
    Then no search result mentions "Eiffel"

  Scenario: Read-only console users cannot change collections
    Given I am signed in to the API as a viewer
    When I try to create a RAG collection
    Then the response status is 403
