@api
Feature: OpenAI-compatible API
  Existing OpenAI SDK clients work against the gateway unchanged.

  Background:
    Given I use the master key

  Scenario: Unary chat completion through the official SDK
    When I ask "eval-chat" "Say hello" with the OpenAI SDK
    Then the SDK reply is "Reply to: Say hello"
    And the SDK reports token usage

  Scenario: Streaming chat completion through the official SDK
    When I stream "eval-chat" "Count to three" with the OpenAI SDK
    Then the streamed text is "Reply to: Count to three"

  Scenario: Tool calling through the official SDK
    When I ask "eval-chat" "__tool__ weather in Paris" with the OpenAI SDK offering the "get_weather" tool
    Then the SDK reply calls the "get_weather" tool

  Scenario: Embeddings through the official SDK
    When I embed "hello world" with "text-embedding-3-small" using the OpenAI SDK
    Then I get 1 embedding of 256 dimensions

  Scenario: Model catalogue
    When I list models with the OpenAI SDK
    Then the model list includes "eval-chat"
    And the model list includes "eval-router"

  Scenario: Unknown models are rejected with an OpenAI error envelope
    When I send a chat request for model "no-such-model" saying "hi"
    Then the response status is 404
    And the error code is "not_found"

  Scenario: Requests without a key are rejected
    When I send a chat request without credentials
    Then the response status is 401
