@ui
Feature: RAG collections and MCP servers in the console

  Scenario: Build and query a RAG collection
    Given I am signed in to the console as the admin
    And I go to "/rag"
    When I create a RAG collection named "<unique>"
    And I paste a document titled "volcanoes" with text "Mount Etna is an active volcano on the island of Sicily."
    Then the collection shows 1 document
    When I run a retrieval test for "Etna volcano Sicily"
    Then a retrieval result mentions "Mount Etna"

  Scenario: Register an MCP server
    Given I am signed in to the console as the admin
    And I go to "/mcp"
    When I add an HTTP MCP server named "<unique>" at the fake MCP URL
    Then the MCP server "<same>" is listed as "healthy" with 3 tools
