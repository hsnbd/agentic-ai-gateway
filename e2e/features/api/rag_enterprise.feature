@api
Feature: Enterprise RAG
  Collections belong to the team (or key) that created them, shared
  collections are read-only to applications, and lost vectors can be rebuilt.

  Scenario: Another team cannot see or search a team's collection
    Given a RAG collection owned by team "alpha" with a document about "Eiffel Tower"
    When a key from team "beta" searches that collection
    Then the response status is 404
    When a key from team "alpha" searches that collection for "Eiffel Tower"
    Then the response status is 200

  Scenario: Applications cannot change a shared collection
    Given I use the master key
    And a RAG collection with the documents:
      | title | content                    |
      | faq   | Refunds take five days.    |
    And a virtual key with no limits
    When I add a document to that collection
    Then the response status is 403

  Scenario: A collection can be reindexed from stored chunks
    Given I use the master key
    And a RAG collection with the documents:
      | title | content                    |
      | faq   | Refunds take five days.    |
    When I reindex that collection
    Then the response status is 202
    And the collection index is in sync with 1 chunk
