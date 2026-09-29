@api
Feature: Reliability: retries, failover, and error handling
  The eval catalogue's primary "eval-chat" deployment points at a dead port,
  so every request has to fail over to the healthy fallback.

  Background:
    Given I use the master key

  Scenario: Failover from a dead primary deployment
    When I send a chat request for model "eval-chat" saying "hello"
    Then the response status is 200
    And the response header "X-Gateway-Deployment" is "openai/eval-chat#2"
    And the reply is "Reply to: hello"

  Scenario: Transient upstream failures are retried
    When I send a chat request for model "eval-router" saying "__flaky__ <unique>"
    Then the response status is 200

  Scenario: A request that fails everywhere returns a structured error
    When I send a chat request for model "eval-chat" saying "__fail__"
    Then the response status is 502
    And the error code is "all_providers_failed"

  Scenario: The failure is recorded in the request log
    When I send a chat request for model "eval-chat" saying "__fail__ <unique>"
    Then the request log has an "error" entry for model "eval-chat"
