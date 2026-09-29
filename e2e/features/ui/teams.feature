@ui
Feature: Teams in the console

  Background:
    Given I am signed in to the console as the admin
    And I go to "/teams"

  Scenario: Create, edit, inspect, and delete a team
    When I create a team named "<unique>" with a budget of 25 USD
    Then the team "<same>" is listed with budget "$0.00 / $25.00"
    When I rename the team "<same>" to "<same>-renamed"
    Then the team "<same>-renamed" is listed with budget "$0.00 / $25.00"
    When I open the usage of team "<same>-renamed"
    Then I see "Requests:" in the dialog
    When I close the dialog
    And I delete the team "<same>-renamed"
    Then the team "<same>-renamed" is not listed
