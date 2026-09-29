@api
Feature: Anthropic-compatible API
  Anthropic SDK clients and coding agents work against the gateway unchanged.

  Background:
    Given I use the master key

  Scenario: Messages API through the official SDK
    When I send "Hi Claude" to "eval-chat" with the Anthropic SDK
    Then the Anthropic reply is "Reply to: Hi Claude"

  Scenario: System prompts are accepted
    When I send "Hi" to "eval-chat" with the Anthropic SDK and system prompt "Be terse"
    Then the Anthropic reply contains "Reply to:"

  Scenario: Streaming messages through the official SDK
    When I stream "Tell me a story" to "eval-chat" with the Anthropic SDK
    Then the streamed text is "Reply to: Tell me a story"
