import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import logging
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')

class UFCStatsScraper:
    """
    Working UFC scraper based on actual website structure
    """
    
    def __init__(self, existing_csv_path=None):
        self.base_url = "http://ufcstats.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load existing data if provided
        self.existing_fights = None
        if existing_csv_path:
            try:
                self.existing_fights = pd.read_csv(existing_csv_path)
                self.logger.info(f"Loaded {len(self.existing_fights)} existing fights from {existing_csv_path}")
            except Exception as e:
                self.logger.error(f"Could not load existing data: {e}")
    
    def get_recent_events(self, max_pages=3, only_completed=True):
        """
        Get recent UFC events using the correct CSS classes from debug output
        """
        events = []
        
        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{self.base_url}/statistics/events/completed"
            else:
                url = f"{self.base_url}/statistics/events/completed?page={page}"
            
            try:
                self.logger.info(f"Scraping events page {page}: {url}")
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Based on debug output: look for table with class 'b-statistics__table-events'
                table = soup.find('table', class_='b-statistics__table-events')
                if table:
                    # Based on debug: found 26 'b-statistics__table-row' elements
                    rows = table.find_all('tr', class_='b-statistics__table-row')[1:]  # Skip header
                    
                    self.logger.info(f"Found {len(rows)} event rows on page {page}")
                    
                    for row in rows:
                        try:
                            # Based on debug: found 50 'b-statistics__table-col' elements (2 per row)
                            cols = row.find_all('td', class_='b-statistics__table-col')
                            
                            if len(cols) >= 2:
                                # First column: event name and date
                                name_col = cols[0]
                                location_col = cols[1]
                                
                                # Extract event link and name
                                event_link = name_col.find('a', href=re.compile(r'/event-details/'))
                                if event_link:
                                    event_url = event_link.get('href', '')
                                    event_name = event_link.get_text(strip=True)
                                    
                                    # Extract date from the same column
                                    date_text = name_col.get_text()
                                    # Look for date pattern (Month DD, YYYY)
                                    date_match = re.search(r'[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}', date_text)
                                    event_date = date_match.group(0) if date_match else "Unknown"
                                    
                                    # Extract location
                                    event_location = location_col.get_text(strip=True)
                                    
                                    # Filter out future events if only_completed=True
                                    if only_completed and event_date != "Unknown":
                                        try:
                                            event_datetime = datetime.strptime(event_date, "%b %d, %Y")
                                            current_datetime = datetime.now()
                                            
                                            # Skip events that are in the future
                                            if event_datetime > current_datetime:
                                                self.logger.debug(f"Skipping future event: {event_name} on {event_date}")
                                                continue
                                        except:
                                            # If date parsing fails, include the event anyway
                                            pass
                                    
                                    if event_name and event_url:
                                        events.append({
                                            'name': event_name,
                                            'date': event_date,
                                            'location': event_location,
                                            'url': event_url
                                        })
                                        self.logger.debug(f"Found completed event: {event_name} on {event_date}")
                        
                        except Exception as e:
                            self.logger.debug(f"Error parsing event row: {e}")
                            continue
                
                else:
                    self.logger.warning(f"No events table found on page {page}")
                
                # Add delay to be respectful
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error scraping page {page}: {e}")
                break
        
        # Remove duplicates
        unique_events = []
        seen_names = set()
        for event in events:
            if event['name'] not in seen_names:
                unique_events.append(event)
                seen_names.add(event['name'])
        
        self.logger.info(f"Found {len(unique_events)} unique completed events")
        return unique_events
    
    def get_fight_details(self, event_url):
        """
        Get detailed fight information from an event page
        Based on debug output: 1 fight table, rows with 10 columns each
        Only return fights with actual results (not upcoming fights)
        """
        if not event_url:
            return []
        
        try:
            full_url = urljoin(self.base_url, event_url)
            self.logger.info(f"Scraping fight details from: {full_url}")
            
            response = self.session.get(full_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            fights = []
            
            # Based on debug: look for 'b-fight-details__table'
            fight_table = soup.find('table', class_='b-fight-details__table')
            if fight_table:
                rows = fight_table.find_all('tr')[1:]  # Skip header
                self.logger.info(f"Found {len(rows)} fight rows")
                
                for row in rows:
                    try:
                        cols = row.find_all('td')
                        if len(cols) >= 8:  # Based on debug: 10 columns per fight
                            
                            # Column 0: Result (win/loss indicator)
                            result_col = cols[0]
                            result_text = result_col.get_text(strip=True).lower()
                            
                            # Column 1: Fighter names
                            fighters_col = cols[1]
                            fighter_links = fighters_col.find_all('a', href=re.compile(r'/fighter-details/'))
                            
                            if len(fighter_links) >= 2:
                                fighter1 = fighter_links[0].get_text(strip=True)
                                fighter2 = fighter_links[1].get_text(strip=True)
                                
                                # Skip fights without results (upcoming fights)
                                # Check if fight has actual method/round/time data
                                method = cols[7].get_text(strip=True) if len(cols) > 7 else ""
                                round_num = cols[8].get_text(strip=True) if len(cols) > 8 else ""
                                time = cols[9].get_text(strip=True) if len(cols) > 9 else ""
                                
                                # If method, round, and time are all empty, this is likely an upcoming fight
                                if not method and not round_num and not time:
                                    self.logger.debug(f"Skipping upcoming fight: {fighter1} vs {fighter2}")
                                    continue
                                
                                # Also skip if result column is empty or just whitespace
                                if not result_text or result_text.isspace():
                                    self.logger.debug(f"Skipping fight with no result: {fighter1} vs {fighter2}")
                                    continue
                                
                                # Determine winner
                                winner = "Unknown"
                                
                                # Look for win indicator in result column
                                if 'win' in result_text or 'w' in result_text:
                                    # Check for specific win indicators
                                    win_indicator = result_col.find('i')
                                    if win_indicator:
                                        # If there's an indicator, first fighter won
                                        winner = fighter1
                                    else:
                                        # Check text structure to determine winner
                                        if result_col.get_text().strip().upper().startswith('W'):
                                            winner = fighter1
                                        else:
                                            winner = fighter2
                                elif 'draw' in result_text or 'nc' in result_text or 'no contest' in result_text:
                                    winner = "Draw/NC"
                                
                                # If we still don't have a winner but have fight details, 
                                # try to determine from method/result
                                if winner == "Unknown" and method:
                                    # For completed fights, we should be able to determine a winner
                                    # If not, skip this fight as it might be upcoming
                                    self.logger.debug(f"Could not determine winner for: {fighter1} vs {fighter2}")
                                    continue
                                
                                # Extract other details
                                try:
                                    weight_class = cols[6].get_text(strip=True) if len(cols) > 6 else "Unknown"
                                    
                                    # Clean up method/round/time
                                    method = method.replace('\n', ' ').strip() if method else "Unknown"
                                    round_num = round_num.strip() if round_num else "Unknown"
                                    time = time.strip() if time else "Unknown"
                                    
                                except:
                                    weight_class = "Unknown"
                                    method = "Unknown"
                                    round_num = "Unknown"
                                    time = "Unknown"
                                
                                # Get fight detail URL for more stats
                                fight_detail_url = None
                                detail_links = row.find_all('a', href=re.compile(r'/fight-details/'))
                                if detail_links:
                                    fight_detail_url = detail_links[0].get('href')
                                
                                fight_data = {
                                    'fighter1': fighter1,
                                    'fighter2': fighter2,
                                    'winner': winner,
                                    'weight_class': weight_class,
                                    'method': method,
                                    'round': round_num,
                                    'time': time,
                                    'fight_url': fight_detail_url
                                }
                                
                                fights.append(fight_data)
                                self.logger.debug(f"Parsed completed fight: {fighter1} vs {fighter2} -> {winner}")
                                
                            else:
                                self.logger.debug(f"Could not find 2 fighter links in row")
                                
                    except Exception as e:
                        self.logger.debug(f"Error parsing fight row: {e}")
                        continue
            
            else:
                self.logger.warning(f"No fight details table found for {event_url}")
            
            self.logger.info(f"Extracted {len(fights)} completed fights from event")
            return fights
            
        except Exception as e:
            self.logger.error(f"Error getting fight details from {event_url}: {e}")
            return []
    
    def scrape_recent_fights(self, max_events=5):
        """
        Main method to scrape recent fights
        """
        self.logger.info("Starting UFC scraping process...")
        
        # Get recent events
        events = self.get_recent_events(max_pages=2)
        
        if not events:
            self.logger.error("No events found!")
            return pd.DataFrame()
        
        # Limit to max_events
        events = events[:max_events]
        self.logger.info(f"Processing {len(events)} events")
        
        all_fights = []
        
        for i, event in enumerate(events):
            self.logger.info(f"Processing event {i+1}/{len(events)}: {event['name']}")
            
            # Get fights from this event
            fights = self.get_fight_details(event['url'])
            
            for fight in fights:
                # Add event information
                fight['event_name'] = event['name']
                fight['event_date'] = event['date']
                fight['event_location'] = event['location']
                
                all_fights.append(fight)
            
            # Add delay between events
            time.sleep(3)
        
        # Convert to DataFrame
        new_fights_df = pd.DataFrame(all_fights)
        
        self.logger.info(f"Scraped {len(new_fights_df)} total fights from {len(events)} events")
        
        return new_fights_df
    
    def find_missing_fights(self, new_fights_df):
        """
        Compare scraped fights with existing dataset to find missing fights
        """
        if self.existing_fights is None:
            self.logger.warning("No existing fights loaded - returning all scraped fights as 'missing'")
            return new_fights_df
        
        if new_fights_df.empty:
            self.logger.warning("No new fights to compare")
            return pd.DataFrame()
        
        # Create a simple comparison key using fighter names and event
        def create_fight_key(row):
            fighter1 = str(row.get('fighter1', '')).strip().lower()
            fighter2 = str(row.get('fighter2', '')).strip().lower()
            event = str(row.get('event_name', '')).strip().lower()
            
            # Sort fighter names to handle order differences
            fighters = tuple(sorted([fighter1, fighter2]))
            return f"{fighters[0]}_{fighters[1]}_{event}"
        
        # Create keys for new fights
        new_fights_df = new_fights_df.copy()
        new_fights_df['fight_key'] = new_fights_df.apply(create_fight_key, axis=1)
        
        # Create keys for existing fights (try different column names)
        existing_keys = set()
        for _, row in self.existing_fights.iterrows():
            # Try to create key from existing data (adapt to your CSV structure)
            existing_row = {
                'fighter1': row.get('fighter1', ''),
                'fighter2': row.get('fighter2', ''),
                'event_name': row.get('event_name', '')
            }
            key = create_fight_key(existing_row)
            existing_keys.add(key)
        
        # Find missing fights
        missing_fights = new_fights_df[~new_fights_df['fight_key'].isin(existing_keys)]
        missing_fights = missing_fights.drop('fight_key', axis=1)
        
        self.logger.info(f"Found {len(missing_fights)} missing fights out of {len(new_fights_df)} scraped fights")
        
        return missing_fights
    
    def save_results(self, fights_df, filename=None):
        """
        Save scraped fights to CSV
        """
        if fights_df.empty:
            self.logger.warning("No fights to save")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ufc_scraped_fights_{timestamp}.csv"
        
        fights_df.to_csv(filename, index=False)
        self.logger.info(f"Saved {len(fights_df)} fights to {filename}")
        
        return filename
    
    def create_test_predictions(self, missing_fights_df, predictor):
        """
        Create predictions for missing fights to test model accuracy
        """
        if missing_fights_df.empty:
            self.logger.warning("No missing fights to create predictions for")
            return pd.DataFrame()
        
        predictions = []
        
        for _, fight in missing_fights_df.iterrows():
            try:
                fighter1 = fight.get('fighter1', '')
                fighter2 = fight.get('fighter2', '')
                actual_winner = fight.get('winner', '')
                
                if not fighter1 or not fighter2:
                    continue
                
                # Search for fighters in predictor database
                f1_matches = predictor.search_fighters(fighter1)
                f2_matches = predictor.search_fighters(fighter2)
                
                if f1_matches and f2_matches:
                    # Use best match
                    f1_name = f1_matches[0]
                    f2_name = f2_matches[0]
                    
                    # Make prediction
                    prediction = predictor.predict_fight(f1_name, f2_name)
                    
                    # Determine if prediction was correct
                    predicted_winner = prediction['predicted_winner']
                    
                    # Map actual winner to predicted fighter names
                    actual_mapped = None
                    if actual_winner and actual_winner != "Unknown" and actual_winner != "Draw/NC":
                        if actual_winner.lower() in fighter1.lower() or fighter1.lower() in actual_winner.lower():
                            actual_mapped = f1_name
                        elif actual_winner.lower() in fighter2.lower() or fighter2.lower() in actual_winner.lower():
                            actual_mapped = f2_name
                    
                    correct_prediction = (predicted_winner == actual_mapped) if actual_mapped else None
                    
                    predictions.append({
                        'event_name': fight.get('event_name', ''),
                        'event_date': fight.get('event_date', ''),
                        'original_fighter1': fighter1,
                        'original_fighter2': fighter2,
                        'mapped_fighter1': f1_name,
                        'mapped_fighter2': f2_name,
                        'actual_winner_original': actual_winner,
                        'actual_winner_mapped': actual_mapped,
                        'predicted_winner': predicted_winner,
                        'fighter1_probability': prediction['fighter1_win_probability'],
                        'fighter2_probability': prediction['fighter2_win_probability'],
                        'confidence': prediction['confidence'],
                        'correct_prediction': correct_prediction,
                        'weight_class': fight.get('weight_class', ''),
                        'method': fight.get('method', ''),
                        'round': fight.get('round', '')
                    })
                    
            except Exception as e:
                self.logger.warning(f"Could not create prediction for {fighter1} vs {fighter2}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        self.logger.info(f"Created {len(predictions_df)} test predictions")
        
        return predictions_df


def quick_test():
    """
    Quick test function to verify the scraper works
    """
    print("Testing Fixed UFC Stats Scraper...")
    print("=" * 50)
    
    # Initialize scraper
    scraper = UFCStatsScraper(existing_csv_path="data/complete_ufc_data.csv")
    
    # Test getting events
    print("1. Testing event scraping...")
    events = scraper.get_recent_events(max_pages=1)
    
    if events:
        print(f"✓ Found {len(events)} events")
        for i, event in enumerate(events[:5]):
            print(f"  {i+1}. {event['name']} ({event['date']}) - {event['location']}")
        
        # Test getting fight details from first event
        if events[0]['url']:
            print(f"\n2. Testing fight details from: {events[0]['name']}")
            fights = scraper.get_fight_details(events[0]['url'])
            
            if fights:
                print(f"✓ Found {len(fights)} fights")
                for i, fight in enumerate(fights[:5]):
                    print(f"  {i+1}. {fight['fighter1']} vs {fight['fighter2']} -> Winner: {fight['winner']}")
                    print(f"      Method: {fight['method']}, Round: {fight['round']}")
            else:
                print("✗ No fights found")
        
        # Test full scraping
        print(f"\n3. Testing full scraping process...")
        all_fights = scraper.scrape_recent_fights(max_events=2)
        
        if not all_fights.empty:
            print(f"✓ Successfully scraped {len(all_fights)} total fights")
            
            # Show sample
            print("\nSample of scraped fights:")
            for _, fight in all_fights.head(3).iterrows():
                print(f"  {fight['event_name']}: {fight['fighter1']} vs {fight['fighter2']} -> {fight['winner']}")
            
            # Save results
            filename = scraper.save_results(all_fights)
            if filename:
                print(f"✓ Saved results to {filename}")
            
            # Test missing fights detection
            missing = scraper.find_missing_fights(all_fights)
            print(f"✓ Found {len(missing)} missing fights")
            
            if not missing.empty:
                print("Missing fights:")
                for _, fight in missing.head(3).iterrows():
                    print(f"  {fight['event_name']}: {fight['fighter1']} vs {fight['fighter2']}")
            
        else:
            print("✗ No fights scraped")
    
    else:
        print("✗ No events found")


if __name__ == "__main__":
    quick_test()