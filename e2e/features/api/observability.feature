@api
Feature: Observability and accounting
  Every request is metered, costed, and inspectable by operators.

  Background:
    Given I use the master key

  Scenario: A request is logged with its cost
    When I send a chat request for model "eval-chat" saying "log me <unique>"
    Then the response status is 200
    And the request log entry for this request is a "success" with a cost above 0

  Scenario: A streamed request is logged too
    When I stream a chat request for model "eval-chat" saying "stream and log me <unique>"
    Then the request log has a streamed "success" entry

  Scenario: Prometheus metrics are exposed
    When I send a chat request for model "eval-chat" saying "metrics <unique>"
    Then the metrics include "aigw_requests_total"
    And the metrics include "aigw_cache_lookups_total"

  Scenario: Spend is attributed to the virtual key
    Given a virtual key with no limits
    When I send a chat request for model "eval-chat" saying "charge me <unique>"
    Then the key's recorded spend is above 0

  Scenario: The readiness probe reports every dependency
    When I check readiness
    Then the response status is 200
    And readiness reports "database", "redis" and "providers" as healthy
