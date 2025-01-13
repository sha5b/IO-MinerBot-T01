from game_ai.core.components.vision.environment_parser import parse_environment

def test_parser():
    """Test the enhanced environment parser with various tactical scenarios"""
    test_cases = [
        {
            "name": "High Threat Scenario",
            "data": {
                "environment": {
                    "terrain_type": "sparse",
                    "threats": [
                        {"type": "enemy", "distance": 50},
                        {"type": "enemy", "distance": 80},
                        {"type": "enemy", "distance": 150}
                    ],
                    "resources": [
                        {"type": "resource", "distance": 300}
                    ]
                },
                "player": {
                    "position": [100, 100]
                }
            }
        },
        {
            "name": "Resource Rich Area",
            "data": {
                "environment": {
                    "terrain_type": "dense",
                    "threats": [
                        {"type": "enemy", "distance": 400}
                    ],
                    "resources": [
                        {"type": "resource", "distance": 100},
                        {"type": "resource", "distance": 150},
                        {"type": "resource", "distance": 180}
                    ]
                },
                "player": {
                    "position": [200, 200]
                }
            }
        },
        {
            "name": "Balanced Situation",
            "data": {
                "environment": {
                    "terrain_type": "moderate",
                    "threats": [
                        {"type": "enemy", "distance": 250},
                        {"type": "enemy", "distance": 350}
                    ],
                    "resources": [
                        {"type": "resource", "distance": 200},
                        {"type": "resource", "distance": 280}
                    ]
                },
                "player": {
                    "position": [150, 150]
                }
            }
        }
    ]

    print("\nTactical Environment Analysis:")
    print("=" * 50)
    
    for case in test_cases:
        result = parse_environment(case['data'])
        print(f"\n{case['name']}:")
        print(f"Summary: {result['summary']}")
        print(f"Details:")
        print(f"- Threat Level: {result['tactical_details']['threat_level']}")
        print(f"- Resources: {result['tactical_details']['resource_availability']}")
        print(f"- Terrain Advantage: {result['tactical_details']['terrain_advantage']}")
        print(f"- Position Quality: {result['tactical_details']['position_quality']}")
        print(f"Overall Rating: {result['tactical_rating']}/10")
        print("-" * 30)

if __name__ == '__main__':
    test_parser()
