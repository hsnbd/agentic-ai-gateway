@api
Feature: Agentic chat through the ordinary chat API
  Any OpenAI-compatible client can ask the gateway to ground an answer in a
  RAG collection, or to run MCP tools on its behalf, with `aigw` extras.

  Scenario: A chat completion grounded in a RAG collection
    Given I use the master key
    And a RAG collection with the documents:
      | title | content                                                       |
      | paris | Paris is the capital of France and home of the Eiffel Tower.  |
    When I ask "eval-chat" "What is the capital of France?" with the OpenAI SDK grounded in the collection
    Then the SDK reply cites a source mentioning "Eiffel"

  Scenario: The gateway runs an MCP tool for the model
    Given I use the master key
    And the fake MCP server is registered
    When I ask "eval-chat" "__tool__:add please add" with the OpenAI SDK using the registered MCP server
    Then the SDK reply is "Tool result: 5"
    And the SDK reports 1 tool call executed by the gateway
