"""
BarberX Legal Case Management Pro Suite
Automated Research Scheduler - Daily Case Law Updates

Runs automated tasks:
- Daily SCOTUS slip opinion monitoring
- Weekly Third Circuit opinion checks
- Weekly NJ Courts opinion checks
- Daily database exports (JSON/HTML)
- Automated publishing to GitHub Pages
"""
import asyncio
import schedule
import time
from datetime import datetime
from pathlib import Path
import logging

from app.services.supreme_law_service import supreme_law_service


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/research_automation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def daily_scotus_check():
    """Check for new SCOTUS slip opinions daily"""
    logger.info("Starting daily SCOTUS slip opinion check...")
    
    try:
        opinions = await supreme_law_service.fetch_scotus_slip_opinions()
        logger.info(f"Found {len(opinions)} SCOTUS slip opinions")
        
        # TODO: Compare with database and add new opinions
        for opinion in opinions:
            logger.info(f"  - {opinion.docket_number}: {opinion.case_name}")
        
        return True
    except Exception as e:
        logger.error(f"SCOTUS check failed: {e}")
        return False


async def daily_automated_update():
    """Run full automated update (SCOTUS + CourtListener)"""
    logger.info("Starting daily automated research update...")
    
    try:
        update_result = await supreme_law_service.automated_daily_update()
        logger.info(f"Update complete: {update_result.summary}")
        logger.info(f"  - SCOTUS: {update_result.new_scotus_count}")
        logger.info(f"  - 3d Circuit: {update_result.new_third_circuit_count}")
        logger.info(f"  - NJ Courts: {update_result.new_nj_count}")
        
        return update_result
    except Exception as e:
        logger.error(f"Automated update failed: {e}")
        return None


async def daily_export_json():
    """Export database to JSON daily"""
    logger.info("Exporting case law database to JSON...")
    
    try:
        await supreme_law_service.export_to_json("./exports/caselaw.json")
        logger.info("JSON export complete")
        return True
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        return False


async def daily_generate_html():
    """Generate HTML portal daily"""
    logger.info("Generating HTML portal...")
    
    try:
        await supreme_law_service.generate_html_portal("./_site/supreme-law/")
        logger.info("HTML portal generated")
        return True
    except Exception as e:
        logger.error(f"HTML generation failed: {e}")
        return False


async def daily_git_publish():
    """Publish updates to GitHub (if configured)"""
    logger.info("Publishing updates to GitHub...")
    
    try:
        import subprocess
        
        # Stage changes
        subprocess.run(['git', 'add', 'data/', 'exports/', '_site/supreme-law/'], check=True)
        
        # Commit
        commit_msg = f"Automated case law update - {datetime.now().strftime('%Y-%m-%d')}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        # Push
        subprocess.run(['git', 'push'], check=True)
        
        logger.info("Published to GitHub successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git publish failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during git publish: {e}")
        return False


# ============================================================
# SCHEDULE CONFIGURATION
# ============================================================

def run_async_task(coro):
    """Helper to run async function in scheduler"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        return result
    finally:
        loop.close()


def setup_schedules():
    """Configure automated task schedules"""
    
    # Daily tasks at 6:00 AM
    schedule.every().day.at("06:00").do(
        lambda: run_async_task(daily_automated_update())
    )
    
    schedule.every().day.at("06:15").do(
        lambda: run_async_task(daily_export_json())
    )
    
    schedule.every().day.at("06:20").do(
        lambda: run_async_task(daily_generate_html())
    )
    
    schedule.every().day.at("06:30").do(
        lambda: run_async_task(daily_git_publish())
    )
    
    # Additional SCOTUS check at 2:00 PM (opinions often released around 10 AM ET)
    schedule.every().day.at("14:00").do(
        lambda: run_async_task(daily_scotus_check())
    )
    
    logger.info("Automated schedules configured:")
    logger.info("  - 06:00 AM: Full automated update")
    logger.info("  - 06:15 AM: JSON export")
    logger.info("  - 06:20 AM: HTML portal generation")
    logger.info("  - 06:30 AM: Git publish")
    logger.info("  - 02:00 PM: SCOTUS slip opinion check")


def run_scheduler():
    """Run the scheduler loop"""
    logger.info("Starting automated research scheduler...")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Setup schedules
    setup_schedules()
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


# ============================================================
# MANUAL TRIGGERS (for testing)
# ============================================================

async def manual_full_update():
    """Manually trigger full update cycle"""
    logger.info("=== MANUAL FULL UPDATE TRIGGERED ===")
    
    # 1. Automated update
    logger.info("Step 1/4: Running automated update...")
    update_result = await daily_automated_update()
    
    # 2. Export JSON
    logger.info("Step 2/4: Exporting to JSON...")
    await daily_export_json()
    
    # 3. Generate HTML
    logger.info("Step 3/4: Generating HTML portal...")
    await daily_generate_html()
    
    # 4. Git publish (optional)
    logger.info("Step 4/4: Publishing to GitHub...")
    # await daily_git_publish()  # Uncomment to enable auto-commit
    
    logger.info("=== MANUAL UPDATE COMPLETE ===")
    return update_result


if __name__ == "__main__":
    # Run scheduler as standalone process
    run_scheduler()
