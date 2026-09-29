@api
Feature: Semantic cache
  Repeated questions are answered from Redis vector search without calling
  (or paying) a provider.

  Background:
    Given I use the master key

  Scenario: A repeated prompt is a free cache hit
    When I send a deterministic chat request saying "What is the tallest mountain? <unique>"
    Then the response header "X-Gateway-Cache" is "miss"
    When I send a deterministic chat request saying "What is the tallest mountain? <same>"
    Then the response header "X-Gateway-Cache" is "hit"
    And the response header "X-Gateway-Cost-USD" is "0.0"

  Scenario: An unrelated prompt is not served from the cache
    When I send a deterministic chat request saying "Describe the ocean <unique>"
    And I send a deterministic chat request saying "Explain quantum tunnelling <unique>"
    Then the response header "X-Gateway-Cache" is "miss"

  Scenario: Callers can opt out of the cache
    When I send a deterministic chat request saying "Opt out please <unique>"
    And I send a deterministic chat request saying "Opt out please <same>" with caching disabled
    Then the response header "X-Gateway-Cache" is "miss"
