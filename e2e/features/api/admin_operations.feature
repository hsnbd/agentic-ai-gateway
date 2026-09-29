@api
Feature: Operating the gateway through the admin API
  Operators reload configuration, inspect policies, manage the cache, and set
  team budgets without restarting anything.

  Scenario: The model catalogue can be reloaded in place
    When an admin reloads the model configuration
    Then the reloaded catalogue includes "eval-chat"
    And the reloaded catalogue includes "eval-router"

  Scenario: Guardrail policies are listed
    When an admin lists the guardrail policies
    Then the response status is 200
    And the guardrail policy "default" is listed

  Scenario: Invalidating the cache forces a fresh answer
    Given I use the master key
    When I send a deterministic chat request saying "Cache me then forget me <unique>"
    And I send a deterministic chat request saying "Cache me then forget me <same>"
    Then the response header "X-Gateway-Cache" is "hit"
    When an admin invalidates the whole semantic cache
    And I send a deterministic chat request saying "Cache me then forget me <same>"
    Then the response header "X-Gateway-Cache" is "miss"

  Scenario Outline: Malformed admin queries are rejected: <case>
    When <request>
    Then the response status is 422

    Examples:
      | case             | request                                           |
      | unknown window   | an admin asks for the dashboard over "fortnight"  |
      | zero window      | an admin asks for the dashboard over "0h"         |
      | secret log order | an admin asks for request logs ordered by "password" |

  Scenario: A team budget caps every key in the team
    Given a team with a budget of 0.000001 USD
    And a virtual key in that team
    When I send a chat request for model "eval-chat" saying "spend the team budget <unique>"
    Then the response status is 200
    When I send a chat request for model "eval-chat" saying "over the team budget <unique>"
    Then the response status is 402
    And the error code is "budget_exceeded"
    And the team's usage reports 1 request
