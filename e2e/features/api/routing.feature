@api
Feature: Routing strategies
  "eval-router" has two healthy deployments that differ in price, priority,
  and weight, so each strategy makes an observable choice.

  Background:
    Given I use the master key

  Scenario Outline: Per-request routing strategy "<strategy>"
    When I send a chat request for model "eval-router" saying "route me" with routing strategy "<strategy>"
    Then the response status is 200
    And the response header "X-Gateway-Deployment" is "<deployment>"

    Examples:
      | strategy   | deployment            |
      | priority   | openai/eval-router    |
      | least-cost | openai/eval-router#2  |
