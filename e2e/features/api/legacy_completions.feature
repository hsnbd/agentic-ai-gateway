@api
Feature: Legacy completions and token counting
  Older clients still call the text-completions API, and Anthropic SDKs count
  tokens before sending. Both go through the same pipeline as chat.

  Background:
    Given I use the master key

  Scenario: A legacy text completion
    When I request a legacy completion for model "eval-chat" with prompt "Name a colour <unique>"
    Then the response status is 200
    And the completion text is "Reply to: Name a colour <same>"
    And the response has a "X-Gateway-Request-Id" header

  Scenario: A streamed legacy completion
    When I stream a legacy completion for model "eval-chat" with prompt "Stream a colour <unique>"
    Then the response status is 200
    And the streamed completion reads "Reply to: Stream a colour <same>"

  Scenario: A legacy completion needs a text prompt
    When I request a legacy completion for model "eval-chat" with prompt ""
    Then the response status is 200
    And the completion text is "Reply to: "

  Scenario: Counting the tokens of a Messages request
    When I count the tokens of "How many tokens is this sentence?" for model "eval-chat"
    Then the response status is 200
    And the input token count is between 5 and 30

  Scenario: Token counting validates the request like the Messages API
    When I count the tokens of a malformed Messages request
    Then the response status is 400
    And the Anthropic error type is "invalid_request_error"
