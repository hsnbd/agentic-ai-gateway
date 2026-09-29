@ui
Feature: Managing virtual keys in the console

  Background:
    Given I am signed in to the console as the admin
    And I go to "/keys"

  Scenario: Creating a key shows the secret exactly once
    When I create a virtual key named "<unique>"
    Then the new key's secret is shown
    And the secret works against the gateway
    When I dismiss the secret
    Then the key "<same>" is listed as "Active"

  Scenario: Disabling a key
    Given a key named "<unique>" exists
    When I reload the page
    And I disable the key "<same>"
    Then the key "<same>" is listed as "Disabled"

  Scenario: Deleting a key
    Given a key named "<unique>" exists
    When I reload the page
    And I delete the key "<same>"
    Then the key "<same>" is not listed
