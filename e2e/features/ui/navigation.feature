@ui
Feature: Role-based console navigation
  Admins can change the gateway; viewers can only look at it.

  Scenario: An admin sees every section
    Given I am signed in to the console as the admin
    Then the navigation shows "Dashboard, Keys, Teams, Models, Logs, Usage, Guardrails, Cache, RAG, MCP, Playground, Settings"

  Scenario: A viewer sees only read-only sections
    Given I am signed in to the console as a new viewer
    Then the navigation shows "Dashboard, Teams, Models, Logs, Usage, Guardrails, Cache, RAG, MCP"
    And the navigation does not show "Keys, Playground, Settings"

  Scenario: A viewer cannot open admin pages by URL
    Given I am signed in to the console as a new viewer
    When I go to "/keys"
    Then I see the "Dashboard" page

  Scenario: A viewer can browse teams but not create them
    Given I am signed in to the console as a new viewer
    When I go to "/teams"
    Then I see the "Teams" page
    And there is no "Create team" button

  Scenario: A viewer can browse RAG collections but not create them
    Given I am signed in to the console as a new viewer
    When I go to "/rag"
    Then I see the "RAG collections" page
    And there is no "Create collection" button

  Scenario Outline: Every page renders without crashing: <page>
    Given I am signed in to the console as the admin
    When I go to "<path>"
    Then I see the "<page>" page
    And the page shows no error

    Examples:
      | path        | page             |
      | /           | Dashboard        |
      | /keys       | Virtual keys     |
      | /teams      | Teams            |
      | /models     | Models & routing |
      | /logs       | Request logs     |
      | /usage      | Usage and costs  |
      | /guardrails | Guardrails       |
      | /cache      | Semantic cache   |
      | /rag        | RAG collections  |
      | /mcp        | MCP servers      |
      | /playground | Playground       |
      | /settings   | Settings         |
